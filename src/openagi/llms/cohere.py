from typing import Any
from langchain_core.messages import HumanMessage
from openagi.exception import OpenAGIException
from openagi.llms.base import LLMBaseModel, LLMConfigModel
from openagi.utils.yamlParse import read_from_env

try:
   from langchain_cohere import ChatCohere
except ImportError:
  raise OpenAGIException("Install langchain groq with cmd `pip install langchain-cohere`")

class CohereConfigModel(LLMConfigModel):
    """Configuration model for Cohere model"""

    cohere_api_key: str
    model_name:str = "command"

class CohereModel(LLMBaseModel):
    """Cohere LLM implementation of the LLMBaseModel.

    This class implements the specific logic required to work with Cohere LLM that runs model locally on CPU.
    """

    config: Any

    def load(self):
        """Initializes the Cohere instance with configurations."""
        self.llm = ChatCohere(
            model = self.config.model_name,
            cohere_api_key = self.config.cohere_api_key,
            temperature = 0.1
        )
        return self.llm

    async def async_load(self):
        return self.load()

    def run(self, input_data: str):
        """Runs the Cohere model with the provided input text.

        Args:
            input_data: The input text to process.

        Returns:
            The response from Cohere LLM service.
        """
        self.load_llm()
        message = self.process_message(input_data=input_data)
        resp = self.llm([message])
        return resp.content
    
    async def async_run(self, input_data: str):
        self.load_llm()
        message = self.process_message(input_data=input_data)
        resp = await self.llm.ainvoke([message])
        return resp.content

    @staticmethod
    def load_from_env_config() -> CohereConfigModel:
        """Loads the Cohere configurations from a YAML file.

        Returns:
            An instance of CohereConfigModel with loaded configurations.
        """
        return CohereConfigModel(
            model_name = read_from_env("COHERE_MODEL",raise_exception=True),
            cohere_api_key = read_from_env("COHERE_API_KEY",raise_exception=True)
        )
