from typing import Tuple, Optional, Iterable, List, Dict

from boto3.dynamodb.conditions import Key

from ..dynamodb import core

ARTICLE_TABLE_NAME = "BP-articleDetail"

ID_VERSION_INDEX = "id-version-index"

# default expressions
DEFAULT_ARTICLE_PROJECTION_EXPRESSION = (
    "id",
    "title",
    "cate",
    "creator_user_id",
)


async def list_articles(
    article_ids: Iterable[str],
    projection_expression: Tuple = DEFAULT_ARTICLE_PROJECTION_EXPRESSION,
) -> List[Optional[dict]]:
    """Retrieves a list of article information from DynamoDB."""

    def to_query_params(x) -> Dict:
        query_param = dict(
            IndexName=ID_VERSION_INDEX,
            KeyConditionExpression=Key("id").eq(x),
        )
        if projection_expression:
            query_param["ProjectionExpression"] = ", ".join(projection_expression)
        return query_param

    query_params = map(to_query_params, article_ids)
    articles = await core.queries(table_name=ARTICLE_TABLE_NAME, query_params=query_params)
    articles = map(lambda x: x[0] if x else None, articles)
    articles = list(articles)
    for article in articles:
        if not article:
            continue
        if "cate" in article:
            article["category"] = article.pop("cate")
        if "id" in article:
            article["article_id"] = article.pop("id")
    return articles
