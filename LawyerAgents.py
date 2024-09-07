from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.coding.local_commandline_code_executor import LocalCommandLineCodeExecutor
from dotenv import load_dotenv
from process import Chatbot
import json
from typing import List, Dict, Any, Annotated
import os

class LawAgent:
    def __init__(self):
        load_dotenv()
        self.chatbot = Chatbot()
        self.folder_path = "Data"
        self.code_executor = LocalCommandLineCodeExecutor(
            work_dir="project",
        )

        self.config_list = [
            {
                "model": os.getenv("model_name"),
                "api_key": os.getenv("api_key"),
                "api_type": "groq",
            }
        ]

        self.llm_config = {
            "timeout": 60,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }

        self.query = "Tell me about the Lawyer how the lawyer can help me in my problems in under 100 words?"

    def termination_msg(self, x):
        return x.get("output", "").rstrip().endswith("TERMINATE")
    
    def preprocessor(self):
        self.file_uploader_agent = AssistantAgent(
            name="file_uploader_agent",
            llm_config=self.llm_config,
            code_execution_config=False,
            human_input_mode="TERMINATE",
            is_termination_msg=self.termination_msg,
            max_consecutive_auto_reply=10,
            description=f"As a File Uploader you need to upload the file and return the path to the user",
        )

        self.Lawyer = AssistantAgent(
            name="Lawyer",
            llm_config=self.llm_config,
            code_execution_config=False,
            human_input_mode="TERMINATE",
            is_termination_msg=self.termination_msg,
            max_consecutive_auto_reply=10,
            description=f"As a Lawyer you need to give the best possible answer to the user query",
        )

        system_message = """
        You are a Legal Compliance Agent. You need to give the best possible answer to the user query.
        # You have get the response from the Lawyer and give the best possible answer to the user query.
        # To provide the answer just focus on these parameters:
        - You should need to provide answer related to given response from the `Lawyer` and the Provide query by the `User`
        - The Response should be more than 300 words
        - The Response should be in the Legal Document Format
        - The Legal Document Format should be like this:
            - "Response from the Lawyer in the Legal Document Format",
            - Lawyer Name: "Surya Pratap",
            - Lawyer Email: "suryapratap@gmail.com",
            - Lawyer Firm Name: "Surya Law Firm",
            - Lawyer Firm Address: "1234 Main Street, Delhi, India",
            - Lawyer Phone Number: "+91 1234567890",
            - Lawyer Bar Council Registration Number: "1234567890",
            - Lawyer Bar Council Registration State: "Delhi",
            - Lawyer Bar Council Registration Country: "India",
            - Lawyer Bar Council Registration Pin Code: "110081",

        # Note:
        - You should not use any other information then provided in the `source` and `information`
        - You should not provide any false information
        - You should not provide any biased answer
        - You should not provide any answer related to any other query
        - You should not provide any answer related to any other query
        """

        self.legal_compliance_agent = ConversableAgent(
            name="legal_compliance_agent",
            llm_config=self.llm_config,
            code_execution_config=False,
            human_input_mode="TERMINATE",
            is_termination_msg=self.termination_msg,
            max_consecutive_auto_reply=10,
            description=f"As a Legal Compliance Agent you need to give the best possible answer to the user query",
            system_message=system_message,
        )

        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            system_message="As a Human Admin, you goal is to Take the Input File Path and Query from the Boss and Get the Result",
            human_input_mode="TERMINATE",
            llm_config=self.llm_config,
            code_execution_config={"work_dir": "coding", "use_docker": False},
            is_termination_msg=self.termination_msg,
        )

        @self.user_proxy.register_for_execution()
        @self.file_uploader_agent.register_for_llm(description=f"Upload the file to the API and return the path to the user and get back the response")
        def file_uploader_func(folder_path: Annotated[str, "The path of the file to be uploaded"]) -> str:
            self.chatbot.prepare_rag(folder_path)
            return f"File Extracted successfully from the folder: {folder_path}"

        @self.user_proxy.register_for_execution()
        @self.Lawyer.register_for_llm(description=f"Get the response as per the user's query")
        def get_response_func(query: Annotated[str, "The query to be sent to the API"])-> str:
            response = self.chatbot.ask(query)
            return response

        self.upload_manager = GroupChat(
            agents=[self.file_uploader_agent, self.user_proxy],
            messages=[],
            max_round=2,
            speaker_selection_method="round_robin",
        )

        self.response_manager = GroupChat(
            agents=[self.user_proxy, self.Lawyer],
            messages=[],
            max_round=2,
            speaker_selection_method="round_robin",
        )

        self.final_report_manager = GroupChat(
            agents=[self.user_proxy, self.legal_compliance_agent],
            messages=[],
            max_round=2,
            speaker_selection_method="round_robin",
        )

        self.manager1 = GroupChatManager(
            groupchat=self.upload_manager,
            llm_config=self.llm_config,
        )

        self.manager2 = GroupChatManager(
            groupchat=self.response_manager,
            llm_config=self.llm_config,
        )

        self.manager3 = GroupChatManager(
            groupchat=self.final_report_manager,
            llm_config=self.llm_config,
        )

        tasks = [
            "Upload the File and return the path to the user folder path - {folder_path}",
            "Get the response from the lawyer as per the user's query - {query}",
            "Get the final report as per the user's query and the response from the lawyer - {query}"
        ]

        self.final_response = self.user_proxy.initiate_chats(
            [
                {
                    "recipient": self.manager1,
                    "message": tasks[0].format(folder_path=self.folder_path),
                    "silent": True,
                    "summary_method": "reflection_with_llm",
                    "max_turns": 1
                },
                {
                    "recipient": self.manager2,
                    "message": tasks[1].format(query=self.query),
                    "silent": True,
                    "summary_method": "last_msg",
                    "max_turns": 1
                },
                {
                    "recipient": self.manager3,
                    "message": tasks[2].format(query=self.query),
                    "silent": True,
                    "summary_method": "last_msg",
                    "max_turns": 1
                }
            ]
        )

        response_content = self.final_response[2].chat_history[1]
        return response_content, self.legal_compliance_agent
    
    def get_response(self, query: str):
        response = self.legal_compliance_agent.generate_reply(
            messages=[
                {
                    "role": "user",
                    "content": query
                },
            ],
            sender=self.user_proxy
        )
        return response['content']


