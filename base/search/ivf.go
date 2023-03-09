// Copyright 2022 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package search

import (
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/chewxy/math32"
	"github.com/zhenghaoz/gorse/base"
	"github.com/zhenghaoz/gorse/base/heap"
	"github.com/zhenghaoz/gorse/base/log"
	"github.com/zhenghaoz/gorse/base/parallel"
	"github.com/zhenghaoz/gorse/base/task"
	"github.com/zhenghaoz/gorse/config"
	"go.uber.org/atomic"
	"go.uber.org/zap"
	"modernc.org/mathutil"
)

type IVF struct {
	clusters []IVFCluster
	vectors  []Vector
	numProbe int
}

type IVFCluster struct {
	centroid     CentroidVector
	observations []int32
	mu           sync.Mutex
}

func (idx *IVF) Search(q Vector, n int, prune0 bool) (values []int32, scores []float32) {
	cq := heap.NewTopKFilter[int, float32](idx.numProbe)
	for c := range idx.clusters {
		d := idx.clusters[c].centroid.Distance(q)
		cq.Push(c, -d)
	}

	pq := heap.NewPriorityQueue(true)
	clusters, _ := cq.PopAll()
	for _, c := range clusters {
		for _, i := range idx.clusters[c].observations {
			if idx.vectors[i] != q {
				pq.Push(i, q.Distance(idx.vectors[i]))
				if pq.Len() > n {
					pq.Pop()
				}
			}
		}
	}
	pq = pq.Reverse()
	for pq.Len() > 0 {
		value, score := pq.Pop()
		if !prune0 || score < 0 {
			values = append(values, value)
			scores = append(scores, score)
		}
	}
	return
}

func (idx *IVF) MultiSearch(q Vector, terms []string, n int, prune0 bool) (values map[string][]int32, scores map[string][]float32) {
	cq := heap.NewTopKFilter[int, float32](idx.numProbe)
	for c := range idx.clusters {
		d := idx.clusters[c].centroid.Distance(q)
		cq.Push(c, -d)
	}

	// create priority queues
	queues := make(map[string]*heap.PriorityQueue)
	queues[""] = heap.NewPriorityQueue(true)
	for _, term := range terms {
		queues[term] = heap.NewPriorityQueue(true)
	}

	// search with terms
	clusters, _ := cq.PopAll()
	for _, c := range clusters {
		for _, i := range idx.clusters[c].observations {
			if idx.vectors[i] != q {
				vec := idx.vectors[i]
				queues[""].Push(i, q.Distance(vec))
				if queues[""].Len() > n {
					queues[""].Pop()
				}
				for _, term := range vec.Terms() {
					if _, match := queues[term]; match {
						queues[term].Push(i, q.Distance(vec))
						if queues[term].Len() > n {
							queues[term].Pop()
						}
					}
				}
			}
		}
	}

	// retrieve results
	values = make(map[string][]int32)
	scores = make(map[string][]float32)
	for term, pq := range queues {
		pq = pq.Reverse()
		for pq.Len() > 0 {
			value, score := pq.Pop()
			if !prune0 || score < 0 {
				values[term] = append(values[term], value)
				scores[term] = append(scores[term], score)
			}
		}
	}
	return
}

// IVFBuilder builds an IVF index from a set of vectors.
type IVFBuilder struct {
	IVF
	topK                int
	clusteringErrorRate float64
	clusteringEpoch     int
	targetRecall        float32
	testSize            int
	maxProbe            int
	bruteForce          *Bruteforce
	randomGenerator     base.RandomGenerator
	jobsAllocator       *task.JobsAllocator
	task                *task.SubTask
}

// NewIVFBuilder creates a new IVFBuilder.
func NewIVFBuilder(data []Vector, topK int, cfg *config.NeighborsConfig) *IVFBuilder {
	b := &IVFBuilder{
		IVF:                 IVF{vectors: data},
		topK:                topK,
		clusteringErrorRate: cfg.IndexClusteringErrorRate,
		clusteringEpoch:     cfg.IndexClusteringEpoch,
		targetRecall:        cfg.IndexTargetRecall,
		testSize:            cfg.IndexTestSize,
		maxProbe:            cfg.IndexMaxProbe,
		bruteForce:          NewBruteforce(data),
		randomGenerator:     base.NewRandomGenerator(0),
	}
	return b
}

func (b *IVFBuilder) evaluate() float32 {
	testSize := mathutil.Min(b.testSize, len(b.vectors))
	samples := b.randomGenerator.Sample(0, len(b.vectors), testSize)
	var result, count float32
	var mu sync.Mutex
	_ = parallel.Parallel(len(samples), b.jobsAllocator.AvailableJobs(b.task.Parent), func(_, i int) error {
		sample := samples[i]
		expected, _ := b.bruteForce.Search(b.vectors[sample], b.topK, true)
		if len(expected) > 0 {
			actual, _ := b.Search(b.vectors[sample], b.topK, true)
			mu.Lock()
			defer mu.Unlock()
			result += recall(expected, actual)
			count++
		}
		return nil
	})
	if count == 0 {
		return 0
	}
	return result / count
}

func (b *IVFBuilder) clustering() {
	if len(b.vectors) == 0 {
		return
	}

	// initialize clusters
	numClusters := int(math.Sqrt(float64(len(b.vectors))))
	clusters := make([]IVFCluster, numClusters)
	assignments := make([]int, len(b.vectors))
	for i := range b.vectors {
		if !b.vectors[i].IsHidden() {
			c := rand.Intn(numClusters)
			clusters[c].observations = append(clusters[c].observations, int32(i))
			assignments[i] = c
		}
	}
	for c := range clusters {
		clusters[c].centroid = b.vectors[0].Centroid(b.vectors, clusters[c].observations)
	}

	for it := 0; it < b.clusteringEpoch; it++ {
		errorCount := atomic.NewInt32(0)

		// reassign clusters
		nextClusters := make([]IVFCluster, numClusters)
		_ = parallel.Parallel(len(b.vectors), b.jobsAllocator.AvailableJobs(b.task.Parent), func(_, i int) error {
			if !b.vectors[i].IsHidden() {
				nextCluster, nextDistance := -1, float32(math32.MaxFloat32)
				for c := range clusters {
					d := clusters[c].centroid.Distance(b.vectors[i])
					if d < nextDistance {
						nextCluster = c
						nextDistance = d
					}
				}
				if nextCluster != assignments[i] {
					errorCount.Inc()
				}
				nextClusters[nextCluster].mu.Lock()
				defer nextClusters[nextCluster].mu.Unlock()
				nextClusters[nextCluster].observations = append(nextClusters[nextCluster].observations, int32(i))
				assignments[i] = nextCluster
			}
			return nil
		})

		log.Logger().Debug("spatial k means clustering", zap.Int32("error_count", errorCount.Load()))
		if float64(errorCount.Load())/float64(len(b.vectors)) < b.clusteringErrorRate {
			break
		}
		for c := range clusters {
			nextClusters[c].centroid = b.vectors[0].Centroid(b.vectors, nextClusters[c].observations)
		}
		clusters = nextClusters
	}
	b.clusters = clusters
}

func (b *IVFBuilder) Build(recall float32, numEpoch int, prune0 bool, t *task.Task) (idx *IVF, score float32) {
	// clustering
	start := time.Now()
	b.clustering()
	clusteringTime := time.Since(start)
	log.Logger().Info("clustering complete", zap.Duration("time", clusteringTime))

	idx.numProbe = 1
	for i := 0; i < numEpoch; i++ {
		score = b.evaluate()
		log.Logger().Info("evaluate index recall",
			zap.Int("num_probe", idx.numProbe),
			zap.Float32("recall", score))
		if score >= recall {
			return
		} else {
			idx.numProbe <<= 1
		}
	}
	return
}

func (b *IVFBuilder) evaluateTermSearch(idx *IVF, prune0 bool, term string) float32 {
	testSize := mathutil.Min(b.testSize, len(b.vectors))
	samples := b.randomGenerator.Sample(0, len(b.vectors), testSize)
	var result, count float32
	var mu sync.Mutex
	_ = parallel.Parallel(len(samples), idx.jobsAlloc.AvailableJobs(idx.task.Parent), func(_, i int) error {
		sample := samples[i]
		expected, _ := b.bruteForce.MultiSearch(b.vectors[sample], []string{term}, b.topK, prune0)
		if len(expected) > 0 {
			actual, _ := idx.MultiSearch(b.vectors[sample], []string{term}, b.topK, prune0)
			mu.Lock()
			defer mu.Unlock()
			result += recall(expected[term], actual[term])
			count++
		}
		return nil
	})
	return result / count
}

func EstimateIVFBuilderComplexity(cfg config.NeighborsConfig, numPoints int) int {
	// clustering complexity
	complexity := cfg.IndexClusteringEpoch * numPoints * int(math.Sqrt(float64(numPoints)))
	// search complexity
	complexity += numPoints * cfg.IndexTestSize * cfg.IndexMaxProbe
	return complexity
}
