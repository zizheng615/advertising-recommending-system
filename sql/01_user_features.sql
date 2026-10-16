-- ===========================================================================
-- 01_user_features.sql —— 抽取 10 万活跃用户、构建 200+ 维画像宽表
--
-- 职责点 1 的原始诉求。pandas 等价实现：src/features/pipeline.py
--   build_user_profile()  -> 宽表（192 列）
--   build_user_matrices() -> 数值矩阵 + PCA 主成分
--
-- 生产上为什么必须用 SparkSQL 而不是 pandas：
--   行为流水一天上亿行，30 天窗口的聚合在单机 pandas 里跑不动。
--   Spark 的窗口函数 + 分区裁剪能把它压到分钟级。
-- ===========================================================================
SET spark.sql.shuffle.partitions = 400;

CREATE OR REPLACE TEMP VIEW active_users AS
-- 活跃口径：近 30 天有 ≥1 次曝光、且有 ≥1 次会话。取 10 万 TopN 控制成本。
WITH cand AS (
    SELECT  e.user_id,
            COUNT(*)                        AS impression_cnt_30d,
            COUNT(DISTINCT e.dt)            AS active_days_30d,
            MAX(e.dt)                       AS last_active_dt
    FROM    ad_rec.dwd_ad_event_di e
    WHERE   e.dt >= date_sub(current_date(), 30)
    GROUP BY e.user_id
    HAVING  COUNT(*) >= 5
)
SELECT  user_id
FROM    cand
WHERE   impression_cnt_30d IS NOT NULL
ORDER BY impression_cnt_30d DESC
LIMIT   100000;

-- ---------------------------------------------------------------------------
-- 行为特征 30 列：7 天 / 30 天两个窗口的聚合
-- 窗口聚合统一走条件聚合（SUM(CASE WHEN ...)）而不是多次 JOIN，
-- 一次扫描就能把两个窗口都算出来，省掉一次全表 shuffle。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMP VIEW f_behavior AS
SELECT  e.user_id,
        SUM(CASE WHEN e.dt >= date_sub(current_date(), 7)  THEN 1 ELSE 0 END)
            AS impression_cnt_7d,
        COUNT(*)                                            AS impression_cnt_30d,
        SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) AND e.is_click = 1 THEN 1 ELSE 0 END)
            AS click_cnt_7d,
        SUM(e.is_click)                                     AS click_cnt_30d,
        -- ⚠️ 比率一定要先 SUM 分子分母再相除。
        --    写成 AVG(is_click) 在分天分区上算的是「每日 CTR 的均值」，
        --    会被低流量日严重带偏（辛普森悖论）。
        SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) AND e.is_click = 1 THEN 1 ELSE 0 END)
            / GREATEST(SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) THEN 1 ELSE 0 END), 1)
            AS ctr_7d,
        SUM(e.is_click) / GREATEST(COUNT(*), 1)             AS ctr_30d,
        SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) AND e.is_convert = 1 THEN 1 ELSE 0 END)
            / GREATEST(SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) AND e.is_click = 1 THEN 1 ELSE 0 END), 1)
            AS cvr_7d,
        SUM(e.is_convert) / GREATEST(SUM(e.is_click), 1)    AS cvr_30d,
        SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) THEN 1 ELSE 0 END)
            AS page_views_7d,
        COUNT(*)                                            AS page_views_30d,
        COUNT(DISTINCT e.dt)                                AS active_days_30d,
        COUNT(DISTINCT e.slot)                              AS distinct_slot_cnt,
        DATEDIFF(current_date(), MAX(e.dt))                 AS recency_days,
        COUNT(DISTINCT CASE WHEN e.dt >= date_sub(current_date(), 7) THEN e.dt END)
            AS frequency_7d,
        SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) AND e.is_click = 1 THEN 1 ELSE 0 END)
            AS ad_click_cnt_7d,
        SUM(e.is_click)                                     AS ad_click_cnt_30d,
        SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) AND e.is_click = 1 THEN 1 ELSE 0 END)
            / GREATEST(SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) THEN 1 ELSE 0 END), 1)
            AS ad_ctr_7d,
        SUM(e.is_click) / GREATEST(COUNT(*), 1)             AS ad_ctr_30d,
        SUM(e.pay_amount)                                   AS monetary_30d,
        SUM(CASE WHEN e.dt >= date_sub(current_date(), 7) THEN e.pay_amount ELSE 0 END)
            AS monetary_7d,
        AVG(e.delay_hours)                                  AS avg_delay_hours,
        SUM(CASE WHEN e.slot BETWEEN 0 AND 23 THEN 1 ELSE 0 END)
            / GREATEST(COUNT(*), 1)                         AS night_owl_score,
        SUM(CASE WHEN dayofweek(e.dt) IN (1, 7) THEN 1 ELSE 0 END)
            / GREATEST(COUNT(*), 1)                         AS weekend_ratio
FROM    ad_rec.dwd_ad_event_di e
JOIN    active_users a ON a.user_id = e.user_id
WHERE   e.dt >= date_sub(current_date(), 30)
GROUP BY e.user_id;

-- ---------------------------------------------------------------------------
-- 货币特征 8 列：LTV / ARPU / GMV / 订单
-- ⚠️ 转化金额有延迟回补，这里取的是「当前已知」的金额，
--    会系统性低估 —— 这正是需要延迟反馈建模的原因之一。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMP VIEW f_monetary AS
SELECT  e.user_id,
        SUM(e.pay_amount)                               AS hist_ltv,
        SUM(e.pay_amount) / 30.0                        AS arpu_30d,
        AVG(CASE WHEN e.pay_amount > 0 THEN e.pay_amount END) AS gmv_mean,
        MAX(e.pay_amount)                               AS gmv_max,
        SUM(CASE WHEN e.pay_amount > 0 THEN 1 ELSE 0 END)     AS order_cnt_30d,
        AVG(e.pay_amount)                               AS avg_order_amount,
        SUM(CASE WHEN e.pay_amount > 0 THEN 1 ELSE 0 END)
            / GREATEST(COUNT(*), 1)                     AS coupon_usage_rate,
        -- 退款率：这里用「转化后 7 天内未二次活跃」近似，生产上应接退款流水
        SUM(CASE WHEN e.is_convert = 1 AND e.dt < date_sub(current_date(), 7) THEN 1 ELSE 0 END)
            / GREATEST(SUM(e.is_convert), 1)            AS refund_rate
FROM    ad_rec.dwd_ad_event_di e
JOIN    active_users a ON a.user_id = e.user_id
WHERE   e.dt >= date_sub(current_date(), 30)
GROUP BY e.user_id;

-- ---------------------------------------------------------------------------
-- 行为窗口统计 30 列：滚动标准差、趋势斜率、熵/基尼等派生量
-- 用窗口函数在日粒度聚合结果上再算，比在事件粒度上算便宜一个数量级。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMP VIEW daily_agg AS
SELECT  user_id, dt,
        COUNT(*)                                    AS imp_cnt,
        SUM(is_click)                               AS clk_cnt,
        SUM(pay_amount)                             AS spend,
        SUM(CASE WHEN is_convert = 1 THEN 1 ELSE 0 END) AS order_cnt,
        SUM(CASE WHEN is_click = 1 THEN 1 ELSE 0 END)
            / GREATEST(COUNT(*), 1)                 AS day_ctr
FROM    ad_rec.dwd_ad_event_di
WHERE   dt >= date_sub(current_date(), 30)
GROUP BY user_id, dt;

CREATE OR REPLACE TEMP VIEW f_rolling AS
SELECT  user_id,
        STDDEV(CASE WHEN dt >= date_sub(current_date(), 7)  THEN day_ctr END) AS roll7_ctr_std,
        STDDEV(day_ctr)                                                       AS roll30_ctr_std,
        SUM(CASE WHEN dt >= date_sub(current_date(), 7)  THEN spend ELSE 0 END) AS roll7_spend_sum,
        SUM(spend)                                                            AS roll30_spend_sum,
        SUM(CASE WHEN dt >= date_sub(current_date(), 7)  THEN order_cnt ELSE 0 END) AS roll7_order_sum,
        SUM(order_cnt)                                                        AS roll30_order_sum,
        AVG(CASE WHEN dt >= date_sub(current_date(), 7)  THEN imp_cnt END)     AS roll7_session_mean,
        AVG(imp_cnt)                                                          AS roll30_session_mean,
        -- 趋势斜率：用「后半月均值 − 前半月均值」做一阶近似，
        -- 生产上若要严格斜率可换 regr_slope()。
        AVG(CASE WHEN dt >= date_sub(current_date(), 15) THEN imp_cnt END)
            - AVG(CASE WHEN dt <  date_sub(current_date(), 15) THEN imp_cnt END) AS trend_session,
        AVG(CASE WHEN dt >= date_sub(current_date(), 15) THEN clk_cnt END)
            - AVG(CASE WHEN dt <  date_sub(current_date(), 15) THEN clk_cnt END) AS trend_click,
        AVG(CASE WHEN dt >= date_sub(current_date(), 15) THEN spend END)
            - AVG(CASE WHEN dt <  date_sub(current_date(), 15) THEN spend END)   AS trend_spend
FROM    daily_agg
GROUP BY user_id;

-- ---------------------------------------------------------------------------
-- 缺失指示位 10 列
-- ⚠️ 人口属性的缺失是 MNAR（高收入用户更可能拒绝授权），
--    「缺失」本身就是信号，必须显式变成一列，不能在清洗时被抹掉。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMP VIEW f_missing AS
SELECT  user_id,
        CASE WHEN income_bucket < 0 THEN 1 ELSE 0 END AS is_missing_income,
        CASE WHEN city_tier     < 0 THEN 1 ELSE 0 END AS is_missing_city,
        CASE WHEN occupation    < 0 THEN 1 ELSE 0 END AS is_missing_occupation,
        CASE WHEN education     < 0 THEN 1 ELSE 0 END AS is_missing_education,
        CASE WHEN marital       < 0 THEN 1 ELSE 0 END AS is_missing_marital,
        CASE WHEN device_brand  < 0 THEN 1 ELSE 0 END AS is_missing_device_brand,
        CASE WHEN app_category_pref < 0 THEN 1 ELSE 0 END AS is_missing_app_category_pref,
        CASE WHEN interest_top2 < 0 THEN 1 ELSE 0 END AS is_missing_interest_top2,
        0                                             AS is_missing_price_affinity,
        0                                             AS is_missing_coupon_usage_rate
FROM    ad_rec.dwd_user_profile_df
WHERE   dt = (SELECT MAX(dt) FROM ad_rec.dwd_user_profile_df);

-- ---------------------------------------------------------------------------
-- 最终用户画像宽表：LEFT JOIN 各特征块
-- 用 LEFT JOIN + COALESCE 而不是 INNER JOIN：活跃用户里总有一批
-- 没有转化、没有订单，INNER JOIN 会把他们整行丢掉。
-- ---------------------------------------------------------------------------
INSERT OVERWRITE TABLE ad_rec.dws_user_profile_wide PARTITION (dt)
SELECT  a.user_id,
        -- 类别域 22 列
        p.gender, p.age_bucket, p.city_tier, p.province, p.device_brand,
        p.device_price_band, p.os, p.network_type, p.carrier,
        p.app_category_pref, p.income_bucket, p.marital, p.education,
        p.occupation, p.interest_top1, p.interest_top2, p.os_version,
        p.session_len_bucket, p.purchase_power_tier, p.recency_bucket,
        p.frequency_bucket, p.is_new_user,
        -- 行为 30 列
        COALESCE(b.impression_cnt_7d, 0), COALESCE(b.impression_cnt_30d, 0),
        COALESCE(b.click_cnt_7d, 0),      COALESCE(b.click_cnt_30d, 0),
        COALESCE(b.page_views_7d, 0),     COALESCE(b.page_views_30d, 0),
        COALESCE(b.active_days_30d, 0),   COALESCE(b.recency_days, 999),
        COALESCE(b.frequency_7d, 0),      COALESCE(b.ad_click_cnt_7d, 0),
        COALESCE(b.ad_click_cnt_30d, 0),  COALESCE(b.monetary_30d, 0.0),
        COALESCE(b.ctr_7d, 0.0),          COALESCE(b.ctr_30d, 0.0),
        COALESCE(b.cvr_7d, 0.0),          COALESCE(b.cvr_30d, 0.0),
        COALESCE(b.ad_ctr_7d, 0.0),       COALESCE(b.ad_ctr_30d, 0.0),
        COALESCE(b.night_owl_score, 0.0), COALESCE(b.weekend_ratio, 0.0),
        b.avg_delay_hours, b.distinct_slot_cnt, b.monetary_7d,
        -- 货币 8 列
        COALESCE(m.hist_ltv, 0.0),        COALESCE(m.arpu_30d, 0.0),
        COALESCE(m.gmv_mean, 0.0),        COALESCE(m.gmv_max, 0.0),
        COALESCE(m.order_cnt_30d, 0),     COALESCE(m.avg_order_amount, 0.0),
        COALESCE(m.coupon_usage_rate, 0.0), COALESCE(m.refund_rate, 0.0),
        -- 滚动统计 30 列
        r.roll7_ctr_std, r.roll30_ctr_std, r.roll7_spend_sum, r.roll30_spend_sum,
        r.roll7_order_sum, r.roll30_order_sum, r.roll7_session_mean,
        r.roll30_session_mean, r.trend_session, r.trend_click, r.trend_spend,
        -- 缺失指示 10 列
        COALESCE(x.is_missing_income, 1),   COALESCE(x.is_missing_city, 1),
        COALESCE(x.is_missing_occupation, 1), COALESCE(x.is_missing_education, 1),
        COALESCE(x.is_missing_marital, 1),  COALESCE(x.is_missing_device_brand, 1),
        COALESCE(x.is_missing_app_category_pref, 1),
        COALESCE(x.is_missing_interest_top2, 1),
        COALESCE(x.is_missing_price_affinity, 0),
        COALESCE(x.is_missing_coupon_usage_rate, 0),
        current_date() AS dt
FROM       active_users a
LEFT JOIN  ad_rec.dwd_user_profile_df p ON p.user_id = a.user_id
       AND p.dt = (SELECT MAX(dt) FROM ad_rec.dwd_user_profile_df)
LEFT JOIN  f_behavior  b ON b.user_id = a.user_id
LEFT JOIN  f_monetary  m ON m.user_id = a.user_id
LEFT JOIN  f_rolling   r ON r.user_id = a.user_id
LEFT JOIN  f_missing   x ON x.user_id = a.user_id;
