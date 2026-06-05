import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Load .env file locally
load_dotenv()


def create_llm(model_name: str):
    """
    Create Gemini LLM using CrewAI LLM class.
    """
    return LLM(
        model=model_name,
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7
    )


def build_crew(topic: str, llm):
    """
    Build CrewAI agents, tasks, and crew.
    """

    # Agent 1: SEO Strategist
    seo_strategist = Agent(
        role="Senior SEO Strategist",
        goal="Find SEO keywords, understand search intent, and create a strong blog outline.",
        backstory=(
            "You are an experienced SEO strategist. You know how to research keywords, "
            "understand user search intent, and create blog outlines that can rank well."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Agent 2: Content Writer
    content_writer = Agent(
        role="Tech Content Writer",
        goal="Write a clear, engaging, beginner-friendly blog post based on the SEO outline.",
        backstory=(
            "You are a professional technical content writer. You explain technical topics "
            "in simple language using examples, headings, and Markdown formatting."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Agent 3: Chief Editor
    editor = Agent(
        role="Chief Editor",
        goal="Polish the blog post for grammar, readability, structure, and SEO quality.",
        backstory=(
            "You are a senior editor. You improve the final article before publishing by "
            "fixing grammar, improving flow, and making the content professional."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Task 1: SEO Planning
    planning_task = Task(
        description=f"""
        Analyze the blog topic: "{topic}"

        Your task:
        1. Identify the target audience.
        2. Find 5 important SEO keywords.
        3. Explain the search intent.
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

    # Task 2: Blog Writing
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

    # Task 3: Final Editing
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

        Do not include internal comments.
        Give only the final blog article.
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

    return seo_blog_crew


def run_seo_crew(topic: str):
    """
    Main function called from Streamlit frontend.
    It runs the CrewAI workflow using Gemini.
    """

    if not topic or topic.strip() == "":
        return "Please enter a valid blog topic."

    if not os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY is missing. Please add it in Streamlit Secrets."

    # Use lighter model first to avoid high-demand 503 error
    models_to_try = [
        "gemini/gemini-2.5-flash-lite",
        "gemini/gemini-1.5-flash"
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            llm = create_llm(model_name)
            crew = build_crew(topic, llm)

            result = crew.kickoff(
                inputs={
                    "topic": topic
                }
            )

            return str(result)

        except Exception as e:
            last_error = str(e)

            # If Gemini is busy, try next model
            if "503" in last_error or "UNAVAILABLE" in last_error or "high demand" in last_error:
                continue

            # For other errors, stop immediately
            return f"Error while running CrewAI workflow: {last_error}"

    return (
        "Gemini is currently busy or unavailable. "
        "Please try again after some time.\n\n"
        f"Last error: {last_error}"
    )
