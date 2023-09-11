from __future__ import annotations

import asyncio
import atexit
from typing import List, Any, Dict, Optional, Tuple, Iterable, Literal

import aioboto3

from ..configs import settings

TABLE_SPACES = {}

RESOURCE_SPACES = {}

LOCK = asyncio.Lock()


async def get_resource(resource_name: Literal["dynamodb"]):
    if resource_name in RESOURCE_SPACES:
        return RESOURCE_SPACES[resource_name]
    session = aioboto3.Session(region_name=settings.aws_region_name)
    resource = session.resource(resource_name, verify=False)
    dynamo_resource = await resource.__aenter__()
    RESOURCE_SPACES[resource_name] = dynamo_resource
    return dynamo_resource


async def get_table(table_name: str):
    async with LOCK:
        if table_name in TABLE_SPACES:
            return TABLE_SPACES[table_name]
        dynamodb_resource = await get_resource("dynamodb")
        table = await dynamodb_resource.Table(table_name)
        TABLE_SPACES[table_name] = table
        return table


async def fetch(
    table,
    query_kwargs: Dict[str, Any],
    limit: Optional[int] = None,
) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Fetches data from the given table based on the provided query parameters.

    This function queries the specified table using the provided query_kwargs, and
    retrieves the items from the query results. If a limit is specified, it will only
    retrieve up to that many items. The function returns a tuple of the list of items
    retrieved, and the last evaluated key of the query (if there is one).

    Args:
        table: The table to query.
        query_kwargs (Dict[str, Any]): The keyword arguments to pass to the query method.
        limit (int, optional): The maximum number of items to retrieve. Defaults to None.

    Returns:
        Tuple[List[Dict], Optional[Dict]]: The list of items retrieved, and the last evaluated key.
    """
    done = False
    item_key = None
    query_results = []

    while not done:
        if item_key:
            query_kwargs["ExclusiveStartKey"] = item_key

        if limit is not None:
            limit = limit - len(query_results)
            query_kwargs["Limit"] = limit

        response = await table.query(**query_kwargs)
        query_results.extend(response["Items"])
        item_key = response.get("LastEvaluatedKey", None)
        done = item_key is None

        if limit is not None and len(query_results) >= limit:
            break

    return query_results, item_key


async def query(
    table_name: str,
    query_param: Dict[str, Any],
    limit: Optional[int] = None,
) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Performs a query on the specified table and returns the results.

    This function queries the specified table using the provided query_param, and
    retrieves the items from the query results. If a limit is specified, it will only
    retrieve up to that many items. The function returns a tuple of the list of items
    retrieved, and the last evaluated key of the query (if there is one).

    Args:
        table_name (str): The name of the table to query.
        query_param (Dict[str, Any]): The query parameter for the query.
        limit (int, optional): The maximum number of items to retrieve. Defaults to None.

    Returns:
        Tuple[List[Dict], Optional[Dict]]: The list of items retrieved, and the last evaluated key.
    """
    table = await get_table(table_name)
    result = await fetch(table, query_param, limit=limit)
    return result


async def queries(
    table_name: str,
    query_params: Iterable[Dict],
    limit: Optional[int] = None,
) -> List[List[Dict]]:
    """
    Performs multiple queries on the specified table and returns the results.

    This function performs multiple queries on the specified table using the provided query_params,
    and retrieves the items from the query results. If a limit is specified, it will only
    retrieve up to that many items for each query. The function returns a list of list of items retrieved.

    Args:
        table_name (str): The name of the table to query.
        query_params (Iterable[Dict]): The list of query parameter for the query.
        limit (int, optional): The maximum number of items to retrieve for each query. Defaults to None.
    Returns:
        List[List[Dict]]: The list of list of items retrieved.
    """

    table = await get_table(table_name)
    coroutines = map(lambda q: fetch(table, q, limit=limit), query_params)
    results = await asyncio.gather(*coroutines)
    return [result or [] for result, _ in results]


async def put(table_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Put an item in the specified table

    This function put an item in the specified table and returns the response.

    Args:
        table_name (str): The name of the table to put the item in
        item (Dict[str, Any]): The item to put in the table

    Returns:
        Dict[str, Any]: The response of the put_item method
    """
    table = await get_table(table_name)
    response = await table.put_item(Item=item)
    return response


async def update(table_name: str, update_query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an item in the specified table

    This function update an item in the specified table based on the update_query provided and returns the response.

    Args:
        table_name (str): The name of the table to update the item in
        update_query (Dict[str, Any]): The update query to update the item

    Returns:
        Dict[str, Any]: The response of the update_item method
    """
    table = await get_table(table_name)
    response = await table.update_item(**update_query)
    return response


async def delete(table_name: str, delete_query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete an item in the specified table

    This function deletes an item in the specified table based on the delete_query provided and returns the response.

    Args:
        table_name (str): The name of the table to delete the item from
        delete_query (Dict[str, Any]): The query to identify the item to delete

    Returns:
        Dict[str, Any]: The response of the delete_item method
    """
    table = await get_table(table_name)
    response = await table.delete_item(**delete_query)
    return response


async def scan(
    table_name: str,
    query_kwargs: Dict[str, Any],
    scan_all: bool = True,
) -> Tuple[List[Dict], Optional[Dict]]:
    session = aioboto3.Session(region_name=settings.aws_region_name)
    async with session.resource("dynamodb", verify=False) as dynamo_resource:
        table = await dynamo_resource.Table(table_name)

        done = False
        start_key = None
        query_results = []

        while not done:
            if start_key:
                query_kwargs["ExclusiveStartKey"] = start_key
            response = await table.scan(**query_kwargs)
            query_results.extend(response["Items"])
            start_key = response.get("LastEvaluatedKey", None)
            done = start_key is None
            if scan_all is False:
                break

    return query_results, start_key


async def close():
    async with LOCK:
        for resource in RESOURCE_SPACES.values():
            try:
                await resource.__aexit__(None, None, None)
            except Exception as e:
                print(f"Failed to close resource {resource}: {e}")

        RESOURCE_SPACES.clear()
        TABLE_SPACES.clear()


def close_dynamodb_resources_on_exit():
    asyncio.run(close())


atexit.register(close_dynamodb_resources_on_exit)
