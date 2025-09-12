import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from lawgpt.core.config import settings
from langchain_core.prompts import ChatPromptTemplate
from lawgpt.llm.case_summarizer.schema.case_summarizer import CaseSummarizerSchema

logger = logging.getLogger(__name__)
import yaml

class CaseSummarizerAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
        )
        self.llm_with_structured_output = self.llm.with_structured_output(CaseSummarizerSchema)
        self.system_prompt = self.load_yaml_prompt(path="case_summary_prompt", key="SYSTEM_PROMPT")
        self.user_prompt = self.load_yaml_prompt(path="case_summary_prompt", key="USER_PROMPT")

        logger.info(f"CaseSummarizerAgent initialized with prompt_template_length: {len(self.system_prompt)}")

    def load_yaml_prompt(self, path: str, key: str):
        """Load prompt template from YAML file"""
        with open(f"lawgpt/llm/prompts/{path}.yml", "r") as file:
            return yaml.safe_load(file)[key]

    def summarize_case(self, case_details: str):
        """Summarize the case details"""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Create the user prompt by safely replacing the placeholder
        user_message = self.user_prompt.replace("{case_details}", case_details)
        
        # Create messages directly without template parsing
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm_with_structured_output.invoke(messages)
        return response.case_summary

def main():
    case_details = """
    "{Section 11(Ka)——The Code of Criminal Procedure} Section 342 We ﬁnd no reason not to put reliance on this postmortem examination report. The doctor who held post mortem examination on the dead body and prepared this report also has been examined by the prosecution as P.W. 14. This doctor witness also has deposed before the trial court to the effect that during postmortem examination he found some postmortem burns on the dead body and on dissection he found antimortem blood stain in the subcutaneous tissue to the anterolateral side of the neck and did not ﬁnd any sing of inﬂammation in the bum area and that in his opinion the death was due to asphyxia as a result of throttling which was anti- mortem and homicidal in nature.. (7)".
    """
    agent = CaseSummarizerAgent()
    response = agent.summarize_case(case_details)
    print(f"Case Summary: {response}")

if __name__ == "__main__":
    main()