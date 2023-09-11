SELECT * FROM `oheadline.analytics_server_prod.viewListArticle` WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY) = TIMESTAMP("2023-08-31") LIMIT 1000;


WITH PositiveSamples AS (
  SELECT
    user_property.user_id as user_id,
    data.article_id as article_id,
    TRUE as watch_or_not,
    data.determined_at as time
  FROM
    `oheadline.analytics_server_prod.viewDetailArticle`
  WHERE
    TIMESTAMP_TRUNC(_PARTITIONTIME, DAY) = TIMESTAMP("2023-08-31")
),
NegativeSamples AS (
  SELECT
    user_property.user_id as user_id,
    data.article_id as article_id,
    FALSE as watch_or_not,
    data.determined_at as time
  FROM
    `oheadline.analytics_server_prod.viewListArticle`
  WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY) = TIMESTAMP("2023-08-31")
  AND data.article_id NOT IN (SELECT article_id FROM PositiveSamples WHERE user_id = PositiveSamples.user_id)
),
CombinedSamples AS (
  SELECT * FROM PositiveSamples
  UNION ALL
  SELECT * FROM NegativeSamples
),
RankedSamples AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY user_id, watch_or_not ORDER BY time DESC) as rank
  FROM
    CombinedSamples
)
SELECT
  user_id,
  article_id,
  watch_or_not,
  time
FROM
  RankedSamples
WHERE
  rank <= 32
ORDER BY
  user_id,
  watch_or_not DESC,
  time DESC;


WITH PositiveSamples AS (
  SELECT
    user_property.user_id as user_id,
    data.article_id as article_id,
    TRUE as watch_or_not,
    data.determined_at as time,
    ROW_NUMBER() OVER (PARTITION BY user_property.user_id ORDER BY data.determined_at DESC) as pos_rank
  FROM
    `oheadline.analytics_server_prod.viewDetailArticle`
  WHERE
    TIMESTAMP_TRUNC(_PARTITIONTIME, DAY) >= TIMESTAMP("2023-06-01")
),
NegativeSamples AS (
  SELECT
    user_property.user_id as user_id,
    data.article_id as article_id,
    FALSE as watch_or_not,
    data.determined_at as time,
    ROW_NUMBER() OVER (PARTITION BY user_property.user_id ORDER BY data.determined_at DESC) as neg_rank
  FROM
    `oheadline.analytics_server_prod.viewListArticle`
  WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY) >= TIMESTAMP("2023-06-01")
  AND data.article_id NOT IN (SELECT article_id FROM PositiveSamples WHERE user_id = PositiveSamples.user_id)
),
CombinedSamples AS (
  SELECT * FROM PositiveSamples
  UNION ALL
  SELECT * FROM NegativeSamples
),
RankedSamples AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY user_id, watch_or_not ORDER BY time DESC) as rank
  FROM
    CombinedSamples
),
MatchedRanks AS (
  SELECT
    pos.user_id,
    pos.article_id as pos_article,
    neg.article_id as neg_article,
    pos.time as pos_time,
    neg.time as neg_time
  FROM (
    SELECT * FROM RankedSamples WHERE watch_or_not = TRUE
  ) pos
  JOIN (
    SELECT * FROM RankedSamples WHERE watch_or_not = FALSE
  ) neg
  ON pos.user_id = neg.user_id AND pos.rank = neg.rank
)
SELECT
  user_id,
  CASE
    WHEN watch_or_not THEN pos_article
    ELSE neg_article
  END as article_id,
  watch_or_not,
  CASE
    WHEN watch_or_not THEN pos_time
    ELSE neg_time
  END as time
FROM MatchedRanks,
UNNEST([TRUE, FALSE]) as watch_or_not
ORDER BY user_id, watch_or_not DESC, time DESC;
