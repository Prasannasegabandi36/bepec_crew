import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

def run_seo_crew(topic: str):
    """
    Initializes and executes the CrewAI multi-agent system for a given topic.
    """
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # 1. Define Agents
    seo_strategist = Agent(
        role='Senior SEO Strategist',
        goal='Uncover high-potential informational keywords and structure content outlines.',
        backstory="""You are an expert in Search Engine Optimization. 
        You understand search intent (informational vs transactional) and how to structure 
        articles (H1, H2, H3) to rank on Google.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    content_writer = Agent(
        role='Tech Content Writer',
        goal='Craft engaging, informative blog posts based on strict outlines.',
        backstory="""You are a skilled writer who specializes in taking technical SEO outlines 
        and turning them into engaging, human-readable narratives. You avoid fluff and 
        focus on value.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    editor = Agent(
        role='Chief Editor',
        goal='Polish content for clarity, tone, and SEO accuracy.',
        backstory="""You are a strict editor. You ensure the content flows well, 
        contains no grammatical errors, and adheres to the SEO keywords provided 
        by the strategist.""",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    # 2. Define Tasks
    planning_task = Task(
        description=f"""
        1. Analyze the given topic: '{topic}'.
        2. Identify 5 target semantic keywords suitable for informational intent.
        3. Create a comprehensive blog post outline with H1, H2, and H3 headers.
        """,
        expected_output="A structured document containing a list of Keywords and a Detailed Outline.",
        agent=seo_strategist
    )

    writing_task = Task(
        description="""
        Using the outline and keywords provided by the SEO Strategist, write a full blog post.
        - Use Markdown formatting.
        - Ensure the tone is helpful and professional.
        - Naturally integrate the keywords.
        """,
        expected_output="A complete blog post draft in Markdown format, approx 800 words.",
        agent=content_writer,
        context=[planning_task] 
    )

    editing_task = Task(
        description="""
        Review the draft blog post.
        1. Check for flow and readability.
        2. Ensure the keywords from the strategist are present.
        3. Output the final, polished version of the blog post.
        """,
        expected_output="The final, polished blog post in Markdown, ready for publishing.",
        agent=editor,
        context=[planning_task, writing_task]
    )

    # 3. Assemble and Kickoff Crew
    seo_blog_crew = Crew(
        agents=[seo_strategist, content_writer, editor],
        tasks=[planning_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True
    )

    inputs = {'topic': topic}
    result = seo_blog_crew.kickoff(inputs=inputs)
    
    # Return the final string output from the crew
    return str(result)