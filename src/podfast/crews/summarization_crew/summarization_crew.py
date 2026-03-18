from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class SummarizationCrew:
    """Podcast summarization crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def podcast_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["podcast_summarizer"],  # type: ignore[index]
            llm="openai/gpt-4o",
        )

    @task
    def summarize_transcript(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_transcript"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
