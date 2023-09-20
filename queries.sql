WITH InitialPositiveSamples AS (
    SELECT
        user_property.user_id as user_id,
        data.article_id as article_id,
        data.watch_time as watch_time,
        true as has_viewed,
        data.determined_at as determined_at,
        ROW_NUMBER() OVER(PARTITION BY user_property.user_id, data.article_id ORDER BY data.determined_at DESC) as row_num_1
    FROM `oheadline.analytics_server_prod.viewDetailArticle`
    WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY)
    BETWEEN TIMESTAMP("2023-06-20") AND TIMESTAMP("2023-09-20")
),
UniquePositiveSamples AS (
    SELECT
        *,
        ROW_NUMBER() over (PARTITION BY user_id ORDER BY determined_at DESC) as row_num_2
    FROM InitialPositiveSamples
    WHERE row_num_1 = 1
),
UserArticleCounts AS (
    SELECT
        user_id,
        COUNT(article_id) as article_count
    FROM UniquePositiveSamples
    GROUP BY user_id
),
PositiveSamples AS (
    SELECT
        u.*
    FROM UniquePositiveSamples u
    JOIN UserArticleCounts c
    ON u.user_id = c.user_id
    WHERE row_num_2 <= 32
    AND c.article_count > 2
),
InitialNegativeSamples AS (
    SELECT
        user_property.user_id as user_id,
        data.article_id as article_id,
        0 as watch_time,
        false as has_viewed,
        data.determined_at as determined_at,
        ROW_NUMBER() OVER(PARTITION BY user_property.user_id, data.article_id ORDER BY data.determined_at DESC) as row_num_3
    FROM `oheadline.analytics_server_prod.viewListArticle`
    WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY)
    BETWEEN TIMESTAMP("2023-06-20") AND TIMESTAMP("2023-09-20")
),
UniqueNegativeSamples AS (
    SELECT
      *
    FROM InitialNegativeSamples
    WHERE row_num_3 = 1
),
FilteredNegativeSamples AS (
    -- Remove negative samples that are also positive samples using left join
    SELECT
        ns.user_id as user_id,
        ns.article_id as article_id,
        ns.watch_time as watch_time,
        ns.has_viewed as has_viewed,
        ns.determined_at as determined_at,
        ROW_NUMBER() OVER(PARTITION BY ns.user_id ORDER BY ns.determined_at DESC) as row_num_4
    FROM UniqueNegativeSamples ns
    LEFT JOIN PositiveSamples ps
    ON ns.article_id = ps.article_id
    AND ns.user_id = ps.user_id
    WHERE ps.article_id IS NULL
),
NegativeSamples AS (
    SELECT
        user_id,
        article_id,
        watch_time,
        has_viewed,
        determined_at,
        row_num_4
    FROM FilteredNegativeSamples
    WHERE row_num_4 <= 32
),
Article AS(
    SELECT
        id as article_id,
        title
    FROM `oheadline.skylake_prod.article`
    WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY)
    BETWEEN TIMESTAMP("2023-05-20") AND TIMESTAMP("2023-09-20")
)
SELECT
    sample.user_id as user_id,
    sample.article_id as aricle_id,
    article.title as title,
    watch_time as watch_time,
    has_viewed as has_viewed,
    determined_at as determined_at
FROM (
    SELECT
        user_id,
        article_id,
        watch_time,
        has_viewed,
        determined_at
    FROM NegativeSamples
    UNION ALL
    SELECT
        user_id,
        article_id,
        watch_time,
        has_viewed,
        determined_at
    FROM PositiveSamples
) sample
INNER JOIN Article as article
ON sample.article_id = article.article_id
WHERE user_id != ''
ORDER BY user_id, has_viewed DESC
;