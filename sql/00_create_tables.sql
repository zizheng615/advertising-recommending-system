-- ===========================================================================
-- 00_create_tables.sql —— 宽表建模的输入表定义（Hive / SparkSQL 语法）
--
-- 这只是**生产环境的接口说明**，本仓库不执行它。
-- 真正的执行路径是 pandas 等价实现：src/features/pipeline.py。
-- 两者产出同一张宽表，字段名一一对应，见 README 的「SparkSQL ↔ pandas 映射」。
--
-- 分区策略：所有事件表按 dt 天分区。10 万活跃用户 × 30 天的曝光量在
-- 千万级，没有分区的话每次跑特征都要全表扫，成本高一个数量级。
-- ===========================================================================

CREATE DATABASE IF NOT EXISTS ad_rec;

-- ---------------------------------------------------------------------------
-- 用户基础属性表：一天一个全量快照（拉链表）
-- 用快照而不是流水表，是因为特征要「截至某天的状态」，
-- 流水表算 as-of 状态的写法复杂且容易出错。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ad_rec.dwd_user_profile_df (
    user_id             BIGINT      COMMENT '用户 ID',
    gender              INT         COMMENT '性别码，-1 表示未授权',
    age_bucket          INT         COMMENT '年龄段 0..6',
    city_tier           INT         COMMENT '城市线级 0..4，-1 表示缺失',
    province            INT         COMMENT '省份码 0..33',
    device_brand        INT         COMMENT '设备品牌码，-1 表示缺失',
    device_price_band   INT         COMMENT '设备价位段 0..4',
    os                  INT         COMMENT '操作系统码',
    network_type        INT         COMMENT '网络类型码',
    carrier             INT         COMMENT '运营商码',
    app_category_pref   INT         COMMENT '偏好类目 0..14',
    income_bucket       INT         COMMENT '收入分档 0..5，-1 表示拒绝授权',
    marital             INT         COMMENT '婚姻状况，-1 表示缺失',
    education           INT         COMMENT '学历 0..4，-1 表示缺失',
    occupation          INT         COMMENT '职业 0..11，-1 表示缺失',
    interest_top1       INT         COMMENT '一级兴趣 0..29',
    interest_top2       INT         COMMENT '二级兴趣 0..29，-1 表示缺失',
    os_version          INT         COMMENT '系统版本码',
    session_len_bucket  INT         COMMENT '会话时长分档 0..5',
    purchase_power_tier INT         COMMENT '购买力分档 0..4',
    recency_bucket      INT         COMMENT '最近活跃分档 0..5',
    frequency_bucket    INT         COMMENT '活跃频次分档 0..5',
    is_new_user         INT         COMMENT '是否新用户 0/1'
)
USING PARQUET
PARTITIONED BY (dt STRING)
COMMENT '用户属性日快照';

-- ---------------------------------------------------------------------------
-- 用户行为流水（点击/浏览/停留等）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ad_rec.dwd_user_behavior_di (
    user_id     BIGINT,
    event_time  TIMESTAMP,
    event_type  STRING  COMMENT 'page_view / click / video_view / search / collect / share / comment / follow',
    item_id     BIGINT,
    duration_ms BIGINT  COMMENT '停留时长'
)
USING PARQUET
PARTITIONED BY (dt STRING);

-- ---------------------------------------------------------------------------
-- 广告曝光 / 点击 / 转化事件（本项目的核心事实表）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ad_rec.dwd_ad_event_di (
    request_id  STRING   COMMENT '请求 ID，同一曝光全链路唯一',
    user_id     BIGINT,
    ad_id       BIGINT,
    campaign_id BIGINT,
    advertiser_id BIGINT,
    event_time  TIMESTAMP,
    slot        INT      COMMENT '当天的第几个 15min 时段 0..95',
    position_in_page INT COMMENT '页内位置 0..9',
    page_type   INT      COMMENT '页面类型 0..7',
    is_click    INT      COMMENT '是否点击 0/1',
    is_convert  INT      COMMENT '是否转化 0/1（受观察窗口影响，存在假负）',
    delay_hours DOUBLE   COMMENT '点击到转化的间隔小时数',
    pay_amount  DOUBLE   COMMENT '付费金额，0 表示未付费'
)
USING PARQUET
PARTITIONED BY (dt STRING);

-- ---------------------------------------------------------------------------
-- 广告素材维度表
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ad_rec.dim_ad (
    ad_id             BIGINT,
    ad_category       INT,
    creative_type     INT,
    creative_format   INT,
    landing_page_type INT,
    price_band        INT,
    color_tone        INT,
    has_video         INT,
    bid_floor_cpm     DOUBLE COMMENT '底价，元/千次曝光'
)
USING PARQUET;

-- ---------------------------------------------------------------------------
-- 广告主维度表
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ad_rec.dim_advertiser (
    advertiser_id      BIGINT,
    advertiser_quality DOUBLE COMMENT '广告主质量分 0..5',
    industry           INT
)
USING PARQUET;
