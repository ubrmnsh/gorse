// Copyright 2021 gorse Project Authors
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

package data

import (
	"context"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/XSAM/otelsql"
	"github.com/dzwvip/oracle"
	"github.com/go-redis/redis/v9"
	"github.com/juju/errors"
	"github.com/samber/lo"
	"github.com/zhenghaoz/gorse/base/log"
	"github.com/zhenghaoz/gorse/storage"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/x/mongo/driver/connstring"
	"go.opentelemetry.io/contrib/instrumentation/go.mongodb.org/mongo-driver/mongo/otelmongo"
	semconv "go.opentelemetry.io/otel/semconv/v1.12.0"
	"gorm.io/driver/clickhouse"
	"gorm.io/driver/mysql"
	"gorm.io/driver/postgres"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"moul.io/zapgorm2"
)

var (
	ErrUserNotExist = errors.NotFoundf("user")
	ErrItemNotExist = errors.NotFoundf("item")
	ErrNoDatabase   = errors.NotAssignedf("database")
)

// Item stores meta data about item.
type Item struct {
	ItemId     string `gorm:"primaryKey"`
	IsHidden   bool
	Categories []string `gorm:"serializer:json"`
	Timestamp  time.Time
	Labels     []string `gorm:"serializer:json"`
	Comment    string
}

// ItemPatch is the modification on an item.
type ItemPatch struct {
	IsHidden   *bool
	Categories []string
	Timestamp  *time.Time
	Labels     []string
	Comment    *string
}

// User stores meta data about user.
type User struct {
	UserId    string   `gorm:"primaryKey"`
	Labels    []string `gorm:"serializer:json"`
	Subscribe []string `gorm:"serializer:json"`
	Comment   string
}

// UserPatch is the modification on a user.
type UserPatch struct {
	Labels    []string
	Subscribe []string
	Comment   *string
}

// FeedbackKey identifies feedback.
type FeedbackKey struct {
	FeedbackType string `gorm:"column:feedback_type"`
	UserId       string `gorm:"column:user_id"`
	ItemId       string `gorm:"column:item_id"`
}

// Feedback stores feedback.
type Feedback struct {
	FeedbackKey `gorm:"embedded"`
	Timestamp   time.Time `gorm:"column:time_stamp"`
	Comment     string    `gorm:"column:comment"`
}

// SortFeedbacks sorts feedback from latest to oldest.
func SortFeedbacks(feedback []Feedback) {
	sort.Sort(feedbackSorter(feedback))
}

type feedbackSorter []Feedback

func (sorter feedbackSorter) Len() int {
	return len(sorter)
}

func (sorter feedbackSorter) Less(i, j int) bool {
	return sorter[i].Timestamp.After(sorter[j].Timestamp)
}

func (sorter feedbackSorter) Swap(i, j int) {
	sorter[i], sorter[j] = sorter[j], sorter[i]
}

type Database interface {
	Init() error
	Ping() error
	Close() error
	Optimize() error
	Purge() error
	BatchInsertItems(ctx context.Context, items []Item) error
	BatchGetItems(ctx context.Context, itemIds []string) ([]Item, error)
	DeleteItem(ctx context.Context, itemId string) error
	GetItem(ctx context.Context, itemId string) (Item, error)
	ModifyItem(ctx context.Context, itemId string, patch ItemPatch) error
	GetItems(ctx context.Context, cursor string, n int, beginTime *time.Time) (string, []Item, error)
	GetItemFeedback(ctx context.Context, itemId string, feedbackTypes ...string) ([]Feedback, error)
	BatchInsertUsers(ctx context.Context, users []User) error
	DeleteUser(ctx context.Context, userId string) error
	GetUser(ctx context.Context, userId string) (User, error)
	ModifyUser(ctx context.Context, userId string, patch UserPatch) error
	UpdateUserId(ctx context.Context, userId string, newUserId string) error
	GetUsers(ctx context.Context, cursor string, n int) (string, []User, error)
	GetUserFeedback(ctx context.Context, userId string, endTime *time.Time, feedbackTypes ...string) ([]Feedback, error)
	GetUserItemFeedback(ctx context.Context, userId, itemId string, feedbackTypes ...string) ([]Feedback, error)
	DeleteUserItemFeedback(ctx context.Context, userId, itemId string, feedbackTypes ...string) (int, error)
	BatchInsertFeedback(ctx context.Context, feedback []Feedback, insertUser, insertItem, overwrite bool) error
	GetFeedback(ctx context.Context, cursor string, n int, beginTime, endTime *time.Time, feedbackTypes ...string) (string, []Feedback, error)
	GetUserStream(ctx context.Context, batchSize int) (chan []User, chan error)
	GetItemStream(ctx context.Context, batchSize int, timeLimit *time.Time) (chan []Item, chan error)
	GetFeedbackStream(ctx context.Context, batchSize int, beginTime, endTime *time.Time, feedbackTypes ...string) (chan []Feedback, chan error)
}

// Open a connection to a database.
func Open(path, tablePrefix string) (Database, error) {
	var err error
	if strings.HasPrefix(path, storage.MySQLPrefix) {
		name := path[len(storage.MySQLPrefix):]
		// probe isolation variable name
		isolationVarName, err := storage.ProbeMySQLIsolationVariableName(name)
		if err != nil {
			return nil, errors.Trace(err)
		}
		// append parameters
		if name, err = storage.AppendMySQLParams(name, map[string]string{
			"sql_mode":       "'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'",
			isolationVarName: "'READ-UNCOMMITTED'",
			"parseTime":      "true",
		}); err != nil {
			return nil, errors.Trace(err)
		}
		// connect to database
		database := new(SQLDatabase)
		database.driver = MySQL
		database.TablePrefix = storage.TablePrefix(tablePrefix)
		if database.client, err = otelsql.Open("mysql", name,
			otelsql.WithAttributes(semconv.DBSystemMySQL),
			otelsql.WithSpanOptions(otelsql.SpanOptions{DisableErrSkip: true}),
		); err != nil {
			return nil, errors.Trace(err)
		}
		database.gormDB, err = gorm.Open(mysql.New(mysql.Config{Conn: database.client}), storage.NewGORMConfig(tablePrefix))
		if err != nil {
			return nil, errors.Trace(err)
		}
		return database, nil
	} else if strings.HasPrefix(path, storage.PostgresPrefix) || strings.HasPrefix(path, storage.PostgreSQLPrefix) {
		database := new(SQLDatabase)
		database.driver = Postgres
		database.TablePrefix = storage.TablePrefix(tablePrefix)
		if database.client, err = otelsql.Open("postgres", path,
			otelsql.WithAttributes(semconv.DBSystemPostgreSQL),
			otelsql.WithSpanOptions(otelsql.SpanOptions{DisableErrSkip: true}),
		); err != nil {
			return nil, errors.Trace(err)
		}
		database.gormDB, err = gorm.Open(postgres.New(postgres.Config{Conn: database.client}), storage.NewGORMConfig(tablePrefix))
		if err != nil {
			return nil, errors.Trace(err)
		}
		return database, nil
	} else if strings.HasPrefix(path, storage.ClickhousePrefix) || strings.HasPrefix(path, storage.CHHTTPPrefix) || strings.HasPrefix(path, storage.CHHTTPSPrefix) {
		// replace schema
		parsed, err := url.Parse(path)
		if err != nil {
			return nil, errors.Trace(err)
		}
		if strings.HasPrefix(path, storage.CHHTTPSPrefix) {
			parsed.Scheme = "https"
		} else {
			parsed.Scheme = "http"
		}
		uri := parsed.String()
		database := new(SQLDatabase)
		database.driver = ClickHouse
		database.TablePrefix = storage.TablePrefix(tablePrefix)
		if database.client, err = otelsql.Open("chhttp", uri,
			otelsql.WithAttributes(semconv.DBSystemKey.String("clickhouse")),
			otelsql.WithSpanOptions(otelsql.SpanOptions{DisableErrSkip: true}),
		); err != nil {
			return nil, errors.Trace(err)
		}
		database.gormDB, err = gorm.Open(clickhouse.New(clickhouse.Config{Conn: database.client}), storage.NewGORMConfig(tablePrefix))
		if err != nil {
			return nil, errors.Trace(err)
		}
		return database, nil
	} else if strings.HasPrefix(path, storage.MongoPrefix) || strings.HasPrefix(path, storage.MongoSrvPrefix) {
		// connect to database
		database := new(MongoDB)
		opts := options.Client()
		opts.Monitor = otelmongo.NewMonitor()
		opts.ApplyURI(path)
		if database.client, err = mongo.Connect(context.Background(), opts); err != nil {
			return nil, errors.Trace(err)
		}
		// parse DSN and extract database name
		if cs, err := connstring.ParseAndValidate(path); err != nil {
			return nil, errors.Trace(err)
		} else {
			database.dbName = cs.Database
			database.TablePrefix = storage.TablePrefix(tablePrefix)
		}
		return database, nil
	} else if strings.HasPrefix(path, storage.SQLitePrefix) {
		// append parameters
		if path, err = storage.AppendURLParams(path, []lo.Tuple2[string, string]{
			{"_pragma", "busy_timeout(10000)"},
			{"_pragma", "journal_mode(wal)"},
		}); err != nil {
			return nil, errors.Trace(err)
		}
		// connect to database
		name := path[len(storage.SQLitePrefix):]
		database := new(SQLDatabase)
		database.driver = SQLite
		database.TablePrefix = storage.TablePrefix(tablePrefix)
		if database.client, err = otelsql.Open("sqlite", name,
			otelsql.WithAttributes(semconv.DBSystemSqlite),
			otelsql.WithSpanOptions(otelsql.SpanOptions{DisableErrSkip: true}),
		); err != nil {
			return nil, errors.Trace(err)
		}
		gormConfig := storage.NewGORMConfig(tablePrefix)
		gormConfig.Logger = &zapgorm2.Logger{
			ZapLogger:                 log.Logger(),
			LogLevel:                  logger.Warn,
			SlowThreshold:             10 * time.Second,
			SkipCallerLookup:          false,
			IgnoreRecordNotFoundError: false,
		}
		database.gormDB, err = gorm.Open(sqlite.Dialector{Conn: database.client}, gormConfig)
		if err != nil {
			return nil, errors.Trace(err)
		}
		return database, nil
	} else if strings.HasPrefix(path, storage.RedisPrefix) {
		addr := path[len(storage.RedisPrefix):]
		database := new(Redis)
		database.client = redis.NewClient(&redis.Options{Addr: addr})
		if tablePrefix != "" {
			panic("table prefix is not supported for redis")
		}
		log.Logger().Warn("redis is used for testing only")
		return database, nil
	} else if strings.HasPrefix(path, storage.OraclePrefix) {
		database := new(SQLDatabase)
		database.driver = Oracle
		database.TablePrefix = storage.TablePrefix(tablePrefix)
		if database.client, err = otelsql.Open("oracle", path,
			otelsql.WithAttributes(semconv.DBSystemOracle),
			otelsql.WithSpanOptions(otelsql.SpanOptions{DisableErrSkip: true}),
		); err != nil {
			return nil, errors.Trace(err)
		}
		database.gormDB, err = gorm.Open(oracle.New(oracle.Config{Conn: database.client}), storage.NewGORMConfig(tablePrefix))
		if err != nil {
			return nil, errors.Trace(err)
		}
		return database, nil
	}
	return nil, errors.Errorf("Unknown database: %s", path)
}
