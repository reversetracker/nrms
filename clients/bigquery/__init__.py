from datetime import datetime, timedelta

import pytz
from google.cloud import bigquery
from google.oauth2 import service_account

import directories

# 서비스 계정 키 파일 경로와 스코프 설정
credentials = service_account.Credentials.from_service_account_file(
    directories.bigquery_google_credential,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)


def list_article_view_counts() -> list:
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    seoul_tz = pytz.timezone("Asia/Seoul")
    current_time_seoul = datetime.now(seoul_tz)
    three_days_before = current_time_seoul - timedelta(days=3)
    four_days_before = current_time_seoul - timedelta(days=4)

    sql_query = f"""
SELECT 
    data.article_id as article_id, 
    count(*) as count
FROM `oheadline.analytics_server_prod.viewDetailArticle`
WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY) BETWEEN TIMESTAMP("{four_days_before.strftime('%Y-%m-%d %H:%M:%S')}")
AND TIMESTAMP("{current_time_seoul.strftime('%Y-%m-%d %H:%M:%S')}")
AND event_time.iso_format >= "{three_days_before.isoformat()}"
AND event_time.iso_format <= "{current_time_seoul.isoformat()}"
GROUP BY article_id
"""

    query_job = client.query(sql_query)
    df = query_job.to_dataframe()
    tuple_list = [(row.article_id, row.count) for row in df.itertuples(index=False)]
    return tuple_list
