import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Load environment variables from .env file locally
load_dotenv()

# Gemini LLM setup
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)


def run_seo_crew(topic: str):
    """
    Runs a CrewAI multi-agent workflow to generate an SEO-optimized blog post.
    Input: topic entered by user from Streamlit frontend
    Output: final polished blog post
    """

    if not topic or topic.strip() == "":
        return "Please enter a valid blog topic."

    # Agent 1: SEO Strategist
    seo_strategist = Agent(
        role="Senior SEO Strategist",
        goal="Find high-potential SEO keywords and create a clear blog outline for the given topic.",
        backstory=(
            "You are an experienced SEO strategist who knows how to research keywords, "
            "understand search intent, and structure blog content for better ranking."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Agent 2: Content Writer
    content_writer = Agent(
        role="Tech Content Writer",
        goal="Write a clear, helpful, and engaging blog post using the SEO outline and keywords.",
        backstory=(
            "You are a professional technical content writer who explains complex topics "
            "in simple language with examples and proper Markdown formatting."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Agent 3: Chief Editor
    editor = Agent(
        role="Chief Editor",
        goal="Improve the blog post for clarity, grammar, readability, structure, and SEO quality.",
        backstory=(
            "You are a senior editor who reviews content before publishing. "
            "You improve flow, fix mistakes, and make the final article polished and professional."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Task 1: SEO planning
    planning_task = Task(
        description=f"""
        Analyze the blog topic: "{topic}"

        Your task:
        1. Identify the target audience.
        2. Find 5 important SEO keywords related to this topic.
        3. Understand the search intent.
        4. Create a complete blog outline using:
           - H1 heading
           - H2 headings
           - H3 subheadings
        5. Suggest a short meta description.

        Keep the output clear and structured.
        """,
        expected_output="""
        A structured SEO plan containing:
        - Target audience
        - Search intent
        - 5 SEO keywords
        - Blog outline with H1, H2, and H3 headings
        - Meta description
        """,
        agent=seo_strategist
    )

    # Task 2: Blog writing
    writing_task = Task(
        description=f"""
        Write a complete blog post on the topic: "{topic}"

        Use the SEO plan and outline from the SEO Strategist.

        Requirements:
        1. Use Markdown formatting.
        2. Write a strong introduction.
        3. Explain the topic in simple language.
        4. Add practical examples where needed.
        5. Naturally include the SEO keywords.
        6. Use proper headings and subheadings.
        7. End with a strong conclusion.

        The blog should be helpful, beginner-friendly, and professional.
        """,
        expected_output="""
        A complete Markdown blog post with:
        - Title
        - Introduction
        - Main sections
        - Examples
        - Conclusion
        """,
        agent=content_writer,
        context=[planning_task]
    )

    # Task 3: Final editing
    editing_task = Task(
        description=f"""
        Review and improve the blog post on: "{topic}"

        Your task:
        1. Check grammar and sentence clarity.
        2. Improve readability.
        3. Make sure the article flows naturally.
        4. Confirm SEO keywords are used naturally.
        5. Improve headings if needed.
        6. Output only the final polished blog post.

        Do not include internal comments. Give only the final blog article.
        """,
        expected_output="""
        A final polished SEO-optimized Markdown blog post ready for publishing.
        """,
        agent=editor,
        context=[planning_task, writing_task]
    )

    # Create Crew
    seo_blog_crew = Crew(
        agents=[seo_strategist, content_writer, editor],
        tasks=[planning_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True
    )

    try:
        result = seo_blog_crew.kickoff(
            inputs={
                "topic": topic
            }
        )
        return str(result)

    except Exception as e:
        return f"Error while running CrewAI workflow: {str(e)}"
