from openagi.actions.base import BaseAction
from pydantic import Field
from duckduckgo_search import DDGS


class DuckDuckGoSearch(BaseAction):
    """Search Tool to fetch results from  DuckDuckGo"""

    name: str = Field(
        default_factory=str,
        description="DuckDuckGoSearch Action to search over duckduckgo using the query.",
    )
    description: str = Field(
        default_factory=str,
        description="Search for words, documents, images, videos, news, maps and text translation using the DuckDuckGo.com search engine.",
    )

    query: str = Field(..., description="User query to fetch web search results from DuckDuckGo")

    # region: str = Field(
    #     default="wt-wt",
    #     description=" which part of the region data needs to clustered: wt-wt, us-en, uk-en, ru-ru",
    # )
    # safesearch: str = Field("off", description="on, moderate, off. Defaults to moderate")
    # max_results: int = Field(
    #     default=10, description="Total results to be executed from the search"
    # )
    # backend: str = Field(
    #     default="api",
    #     description="api, html, lite. Defaults to api. This defines from where the data needs to be requested",
    # )

    def execute(self):
        result = DDGS().text(
            self.query,
            # region=self.region,
            # safesearch=self.safesearch,
            max_results=5,
            # backend=self.backend,
        )
        return result