-- ===========================================================================
-- 02_item_features.sql —— 素材侧特征 + 用户侧广告统计
--
-- pandas 等价实现：src/features/pipeline.py::build_item_matrices()
--
-- 素材侧只有 11 列，为什么也值得单独写一段？
-- 因为它是**双塔模型的 item tower 输入**。item tower 必须能在离线把
-- 全量素材编码成向量、灌进向量检索库，所以它的输入里一个用户特征都不能有 ——
-- 这是双塔最容易被写坏的地方。
-- ===========================================================================

CREATE OR REPLACE TABLE ad_rec.dws_ad_feature AS
SELECT  d.ad_id,
        -- -------- 类别域 7 列（item tower 用） --------
        d.ad_category,
        d.creative_type,
        d.creative_format,
        d.landing_page_type,
        d.price_band,
        d.color_tone,
        d.has_video,
        -- -------- 广告主侧 --------
        d.advertiser_id,
        d.campaign_id,
        COALESCE(adv.advertiser_quality, 2.5) AS advertiser_quality,
        adv.industry,
        -- -------- 数值信号 4 列（item tower 用） --------
        -- 全局 CTR / CVR：素材自身的「历史质量」，与用户无关
        COALESCE(s.ad_ctr_global, 0.0)  AS ad_ctr_global,
        COALESCE(s.ad_cvr_global, 0.0)  AS ad_cvr_global,
        COALESCE(d.bid_floor_cpm, 0.0)  AS bid_floor_cpm,
        -- 素材热度：近 7 天曝光量相对大盘的比值。
        -- ⚠️ 必须做**相对**归一，直接用绝对曝光量的话，
        --    大盘整体涨跌会让这个特征整体漂移，模型只能记住训练期的水位。
        COALESCE(s.imp_cnt_7d, 0) / GREATEST(t.total_imp_7d, 1) AS item_popularity
FROM       ad_rec.dim_ad d
LEFT JOIN  ad_rec.dim_advertiser adv ON adv.advertiser_id = d.advertiser_id
LEFT JOIN (
    SELECT  ad_id,
            SUM(is_click) / GREATEST(COUNT(*), 1)   AS ad_ctr_global,
            SUM(is_convert) / GREATEST(SUM(is_click), 1) AS ad_cvr_global,
            SUM(CASE WHEN dt >= date_sub(current_date(), 7) THEN 1 ELSE 0 END) AS imp_cnt_7d
    FROM    ad_rec.dwd_ad_event_di
    WHERE   dt >= date_sub(current_date(), 30)
    GROUP BY ad_id
) s ON s.ad_id = d.ad_id
CROSS JOIN (
    SELECT  COUNT(*) AS total_imp_7d
    FROM    ad_rec.dwd_ad_event_di
    WHERE   dt >= date_sub(current_date(), 7)
) t;

-- ---------------------------------------------------------------------------
-- 用户侧广告统计 6 列（可在用户粒度算出，属于 user tower）
--
-- ⚠️ 下面这几个是**用户 × 素材类目**的交叉统计。它们同时依赖用户和素材，
--    **只能进排序层，绝不能进 user tower** —— 一旦进了 user tower，
--    用户向量就随素材变化了，双塔失去「离线预算、在线内积」的意义。
--    这里放在同一个文件里只是方便对照，实际生产中它们会落在
--    dws_user_ad_context 表，由排序模型消费。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE ad_rec.dws_user_ad_context AS
SELECT  e.user_id,
        AVG(adv.advertiser_quality)                      AS advertiser_quality,
        -- 类目亲和度：该用户在其偏好类目上的 CTR / 整体 CTR
        SUM(CASE WHEN e.is_click = 1 AND d.ad_category = p.app_category_pref THEN 1 ELSE 0 END)
            / GREATEST(SUM(CASE WHEN d.ad_category = p.app_category_pref THEN 1 ELSE 0 END), 1)
            AS category_affinity,
        -- 价格偏好：该用户点击的素材价位段均值，归一化到 [0,1]
        AVG(CASE WHEN e.is_click = 1 THEN d.price_band END) / 7.0 AS price_affinity,
        -- 创意疲劳：近 3 天曝光占比 − 更早 27 天曝光占比，越大说明越疲劳
        (SUM(CASE WHEN e.dt >= date_sub(current_date(), 3)  THEN 1 ELSE 0 END)
         / GREATEST(COUNT(*), 1))
        - (SUM(CASE WHEN e.dt <  date_sub(current_date(), 3) THEN 1 ELSE 0 END)
         / GREATEST(COUNT(*), 1))                        AS creative_fatigue_score,
        -- 竞争指数：同一 slot 内平均同时竞价的广告主数，归一化
        AVG(1.0 / GREATEST(e.competitor_cnt, 1)) * 2.0   AS competition_index,
        -- 该用户活跃时段的流量占比
        COUNT(DISTINCT e.slot) / 96.0                    AS slot_traffic_share
FROM       ad_rec.dwd_ad_event_di e
JOIN       ad_rec.dim_ad d ON d.ad_id = e.ad_id
LEFT JOIN  ad_rec.dim_advertiser adv ON adv.advertiser_id = e.advertiser_id
LEFT JOIN  ad_rec.dwd_user_profile_df p ON p.user_id = e.user_id
       AND p.dt = (SELECT MAX(dt) FROM ad_rec.dwd_user_profile_df)
WHERE      e.dt >= date_sub(current_date(), 30)
GROUP BY   e.user_id;
