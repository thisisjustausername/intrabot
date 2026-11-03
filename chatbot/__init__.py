"""
Override functions from the langchain_mongodb.agent_toolkit.tool module to fix issues with the get_mongodb_schema function.
"""
import os

import json5
import langchain_mongodb.agent_toolkit.tool as mongodb_tool
from langchain_core.caches import BaseCache as BaseCache
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.callbacks import Callbacks as Callbacks
from langchain_core.tools import BaseTool
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.agent_toolkit.tool import (
    BaseMongoDBDatabaseTool,
    ListMongoDBDatabaseTool,
    QueryMongoDBCheckerTool,
    QueryMongoDBDatabaseTool,
    _InfoMongoDBDatabaseToolInput,
)
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from pydantic import BaseModel

with open(os.path.expanduser('~/mhbai/src/mhbai/mongo_db/collection_schemata.jsonc'), 'r') as f:
    collection_schemata = json5.load(f)
with open(os.path.expanduser('~/mhbai/src/mhbai/mongo_db/collection_indices.json'), 'r') as f:
    collection_indices = json5.load(f)

def get_collection_info(self, collection_names: list[str] | None = None) -> str:
    """Get information about specified collections.

    Follows best practices as specified in: Rajkumar et al, 2022
    (https://arxiv.org/abs/2204.00498)

    If `sample_rows_in_collection_info`, the specified number of sample rows will be
    appended to each collection description. This can increase performance as
    demonstrated in the paper.
    """
    all_coll_names = self.get_usable_collection_names()
    if collection_names is not None:
        missing_collections = set(collection_names).difference(all_coll_names)
        if missing_collections:
            raise ValueError(
                f"collection_names {missing_collections} not found in database"
            )
        all_coll_names = collection_names

    colls = []
    for coll in all_coll_names:
        # add schema
        schema = self._get_collection_schema(coll)
        coll_info = f"Database name: {self._db.name}\n"
        coll_info += f"Collection name: {coll}\n"
        coll_info += f"Schema from a sample of documents from the collection:\n{schema.rstrip()}"
        has_extra_info = (
            self._indexes_in_coll_info or self._sample_docs_in_coll_info
        )
        if has_extra_info:
            coll_info += "\n\n/*"
        if self._indexes_in_coll_info:
            coll_info += f"\n{self._get_collection_indexes(coll)}\n"
        # NOTE: MANUALLY ADDED
        coll_info += f"\nSCHEMA: {collection_schemata.get(coll, None)}\n"
        coll_info += f"\nINDICES: {collection_indices.get(coll, None)}\n"

        if has_extra_info:
            coll_info += "*/"
        colls.append(coll_info)
    colls.sort()
    final_str = "\n\n".join(colls)
    return final_str


class InfoMongoDBDatabaseTool(BaseMongoDBDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting metadata about a MongoDB database."""

    name: str = "mongodb_schema"
    description: str = (
        "Get the schema for the specified MongoDB collections."
    )
    args_schema: type[BaseModel] = _InfoMongoDBDatabaseToolInput # type: ignore

    def _run(
        self,
        collection_names: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Get the schema for collections in a comma-separated list
        Allowed collection names are: mhbs, modules, exams."""

        result = self.db.get_collection_info_no_throw(
            [t.strip() for t in collection_names.split(",")]
        )
        print(f"RESULT: {result}")
        return result




def get_tools(self) -> list[BaseTool]:
    """Get the tools in the toolkit."""
    list_mongodb_database_tool = ListMongoDBDatabaseTool(db=self.db)
    info_mongodb_database_tool_description = (
        "Input to this tool is a comma-separated list of collections, output is the "
        "schema and sample rows for those collections. "
        "Be sure that the collectionss actually exist by calling "
        f"{list_mongodb_database_tool.name} first! "
        "Example Input: collection1, collection2, collection3"
    )
    # NOTE: manually blocked embedding_ in tool -> database -> _get_collection_schema
    info_mongodb_database_tool = InfoMongoDBDatabaseTool(
        db=self.db, description=info_mongodb_database_tool_description
    )
    query_mongodb_database_tool_description = (
        "Input to this tool is a detailed and correct MongoDB query, output is a "
        "result from the database. If the query is not correct, an error message "
        "will be returned. If an error is returned, rewrite the query, check the "
        "query, and try again. If you encounter an issue with Unknown column "
        f"'xxxx' in 'field list', use {info_mongodb_database_tool.name} "
        "to query the correct collections fields."
    )
    query_mongodb_database_tool = QueryMongoDBDatabaseTool(
        db=self.db, description=query_mongodb_database_tool_description
    )
    query_mongodb_checker_tool_description = (
        "Use this tool to double check if your query is correct before executing "
        "it. Always use this tool before executing a query with "
        f"{query_mongodb_database_tool.name}!"
    )
    query_mongodb_checker_tool = QueryMongoDBCheckerTool(
        db=self.db, llm=self.llm, description=query_mongodb_checker_tool_description
    )
    return [
        query_mongodb_database_tool,
        info_mongodb_database_tool,
        list_mongodb_database_tool,
        query_mongodb_checker_tool,
    ]

MongoDBDatabaseToolkit.get_tools = get_tools
mongodb_tool.InfoMongoDBDatabaseTool = InfoMongoDBDatabaseTool # type: ignore
MongoDBDatabase.get_collection_info = get_collection_info
