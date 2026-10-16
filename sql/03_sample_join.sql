-- ===========================================================================
-- 03_sample_join.sql —— 曝光样本拼接 + 标签生成
--
-- pandas 等价实现：src/features/pipeline.py::build_impression_matrices()
--                  src/data/splits.py::time_split()
--
-- 这段负责把「用户宽表 × 素材表 × 曝光事实」拼成训练样本。
-- 三个必须做对、做错也不报错的地方，都在下面的注释里标了 ⚠️。
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 第一步：标签生成
--
-- ⚠️ 假负样本是这里产生的，不是模型的问题。
--    转化有延迟（实测长尾能到 720 小时），但数据只能按「截至今天」来打标签。
--    于是一批已经发生、只是还没被观测到的转化被写成了 0。
--    实测假负率（mini 档）：6h 35.2% / 24h 24.5% / 72h 8.7% / 168h 2.8%，
--    而线上实际拿到的标签（24h 窗口 + 末尾截断）是 25.2%
--    —— 比纯 24h 窗口还高，因为最近那批曝光根本没等够 24 小时。
--    数字来源：artifacts/metrics/01_data_*.json 的 false_negative。
--
--    三套标签的用途严格区分：
--      label_w24   —— 线上真实标签，**训练用**
--      label_w168  —— 回补标签，用于评估「如果等更久能好多少」
--      label_oracle—— 无限窗口真值，**只用于评测，绝不进训练**
-- ===========================================================================
CREATE OR REPLACE TEMP VIEW labels AS
SELECT  e.request_id,
        e.user_id,
        e.ad_id,
        e.dt,
        e.is_click,

        -- 线上真实标签：24h 窗口 + 数据末尾截断
        CASE WHEN e.is_convert = 1 AND e.delay_hours <= 24 THEN 1 ELSE 0 END AS y_conv_w24,
        CASE WHEN e.is_convert = 1 AND e.delay_hours <= 168 THEN 1 ELSE 0 END AS y_conv_w168,

        -- 观测时长：从曝光到数据拉取时刻。末尾 7 天的曝光等不满 168 小时。
        -- ⚠️ 这一列是延迟反馈模型（DFM）的关键输入。
        --    如果它退化成一个常数，DFM 的修正项就完全没有区分度，
        --    而训练照样收敛、指标照样出 —— 属于最危险的那类静默错误。
        LEAST(168.0,
              (unix_timestamp(current_timestamp()) - unix_timestamp(e.event_time)) / 3600.0
        ) AS elapsed_h
FROM    ad_rec.dwd_ad_event_di e
WHERE   e.dt >= date_sub(current_date(), 30);

-- ---------------------------------------------------------------------------
-- 第二步：曝光样本 × 用户宽表 × 素材特征
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMP VIEW samples AS
SELECT  l.request_id,
        l.dt,
        l.user_id,
        l.ad_id,

        -- 上下文类别域（5 列）
        e.slot                                              AS slot,
        CAST(e.slot / 4 AS INT)                             AS hour,
        dayofweek(e.dt)                                     AS dow,
        CAST(e.slot / 24 AS INT)                            AS time_slot,
        e.position_in_page,
        e.page_type,

        -- 跨域特征（**只能进排序层**）
        -- ⚠️ 这两个字段同时依赖用户和素材。放进 user tower 会让用户向量
        --    随素材变化，双塔失去在线内积的意义 —— 这是最常见的实现错误。
        u.category_affinity * a.ad_ctr_global               AS user_x_category_ctr,
        u.price_affinity    * a.ad_cvr_global               AS user_x_creative_cvr,
        a.ad_ctr_global,
        a.ad_cvr_global,
        a.bid_floor_cpm,
        CASE WHEN a.ad_category = p.app_category_pref THEN 1 ELSE 0 END AS user_x_category_hit,

        -- 标签
        l.is_click                                          AS y_click,
        l.y_conv_w24,
        l.y_conv_w168,
        l.elapsed_h
FROM        labels l
JOIN        ad_rec.dwd_ad_event_di e ON e.request_id = l.request_id
LEFT JOIN   ad_rec.dws_ad_feature  a ON a.ad_id = l.ad_id
LEFT JOIN   ad_rec.dws_user_ad_context u ON u.user_id = l.user_id
LEFT JOIN   ad_rec.dwd_user_profile_df p ON p.user_id = l.user_id
       AND p.dt = (SELECT MAX(dt) FROM ad_rec.dwd_user_profile_df);

-- ---------------------------------------------------------------------------
-- 第三步：1:5 负采样（只对训练集做！）
--
-- ⚠️ 采样必须在**时间切分之后**、且只在训练集上做。
--    对整个样本做负采样会让测试集也变形，评测指标失去意义；
--    跨切分做近邻插值（SMOTE）更糟 —— 会造出未来样本的近邻，时序泄漏。
--
-- ⚠️ 采样完必须把先验改回来：logit += ln(β)，β = 负样本保留比例 = 0.2。
--    漏掉这一步，AUC 完全不受影响（排序没变），但概率会整体高估约 4 倍，
--    下游 eCPM 跟着错 4 倍。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMP VIEW train_neg_sampled AS
SELECT * FROM (
    SELECT  s.*,
            ROW_NUMBER() OVER (PARTITION BY s.y_conv_w24 ORDER BY rand(42)) AS rn
    FROM    samples s
    WHERE   s.dt < date_sub(current_date(), 6)      -- 训练集：更早的 24 天
) t
WHERE   y_conv_w24 = 1
   OR   rn <= 5 * (SELECT COUNT(*) FROM samples WHERE dt < date_sub(current_date(), 6) AND y_conv_w24 = 1)
        / GREATEST((SELECT COUNT(*) FROM samples WHERE dt < date_sub(current_date(), 6) AND y_conv_w24 = 0), 1)
        * (SELECT COUNT(*) FROM samples WHERE dt < date_sub(current_date(), 6) AND y_conv_w24 = 0)
        / 5;

-- 更直观的等价写法（生产上推荐用这个，可读性好得多）：
--   WITH pos AS (SELECT * FROM samples WHERE dt < ... AND y_conv_w24 = 1),
--        neg AS (SELECT * FROM samples WHERE dt < ... AND y_conv_w24 = 0
--                DISTRIBUTE BY rand(42) SORT BY rand(42) LIMIT <5 * |pos|>)
--   SELECT * FROM pos UNION ALL SELECT * FROM neg;

-- ---------------------------------------------------------------------------
-- 第四步：时间切分（70% / 15% / 15%）
-- ⚠️ 必须按天整组切，不能随机切。
--    随机切会让同一用户、同一天的曝光同时出现在训练和测试集里，
--    模型记住 user_id 就等于作弊，离线 AUC 会虚高十几个点。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE ad_rec.dws_train_valid_test AS
SELECT  s.*,
        CASE
            WHEN s.dt <  date_sub(current_date(), 6)  THEN 'train'
            WHEN s.dt <  date_sub(current_date(), 3)  THEN 'valid'
            ELSE 'test'
        END AS split_tag
FROM    samples s;

-- ---------------------------------------------------------------------------
-- 验收查询：跑完上面几步，用这几条 SQL 做体检
-- ---------------------------------------------------------------------------

-- 1) 基率是否落在预期（CTR 5%、CVR|点击 6%）
-- SELECT split_tag,
--        AVG(y_click)                                  AS ctr,
--        SUM(y_conv_w168) / GREATEST(SUM(y_click), 1)  AS cvr,
--        COUNT(*)                                      AS n
-- FROM   ad_rec.dws_train_valid_test GROUP BY split_tag;

-- 2) 假负率随窗口的单调曲线（延迟问题的量化）
-- SELECT 1 - SUM(y_conv_w24)  / SUM(y_conv_w168) AS fn_24h,
--        1 - SUM(y_conv_w168) / SUM(oracle)      AS fn_168h
-- FROM   ad_rec.dws_train_valid_test;

-- 3) SRM 检查：分流后两组样本量是否符合预期比例（p < 0.001 说明分流有 bug）
-- SELECT  group_id, COUNT(*) FROM ad_rec.dws_ab_assignment GROUP BY group_id;
