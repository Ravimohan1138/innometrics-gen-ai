import os

from huggingface_hub import InferenceClient

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# =====================================================
# TOKENS
# =====================================================

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxxxx"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"

os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["LANGCHAIN_PROJECT"] = "AI Resume Screening System"

# =====================================================
# HUGGINGFACE CLIENT
# =====================================================

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

# =====================================================
# CUSTOM LLM FUNCTION
# =====================================================

def hf_generate(prompt):

    prompt = str(prompt)

    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=256
    )

    return response.choices[0].message.content
# =====================================================
# LANGCHAIN LLM WRAPPER
# =====================================================

llm = RunnableLambda(hf_generate)

# =====================================================
# JOB DESCRIPTION
# =====================================================

job_description = """
We are hiring a Data Scientist.

Required Skills:
- Python
- Machine Learning
- SQL
- Data Analysis
- Pandas
- NumPy
- Deep Learning

Preferred Tools:
- TensorFlow
- Scikit-learn
- Power BI

Experience:
2+ years experience in Data Science projects.
"""

# =====================================================
# RESUMES
# =====================================================

strong_resume = """
John Doe

Skills:
Python, Machine Learning, SQL,
Pandas, NumPy, TensorFlow,
Scikit-learn, Deep Learning,
Power BI

Experience:
3 years working as Data Scientist.

Projects:
Built ML prediction systems
using Python and TensorFlow.
"""

average_resume = """
Jane Smith

Skills:
Python, SQL, Pandas,
Data Visualization

Experience:
1 year internship experience.

Projects:
Worked on analytics dashboard project.
"""

weak_resume = """
Alex Brown

Skills:
MS Word, Excel, Communication

Experience:
Fresher

Projects:
Created college presentations.
"""

# =====================================================
# PROMPTS
# =====================================================

extract_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
You are an AI Resume Analyzer.

Extract:
1. Skills
2. Tools
3. Experience

IMPORTANT RULES:
- Do NOT assume skills not present.
- Return only resume information.

Resume:
{resume}
"""
)

match_prompt = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="""
Compare the resume with the job description.

Job Description:
{job_description}

Resume:
{resume}

Provide:
1. Matching skills
2. Missing skills
3. Experience comparison

IMPORTANT:
Do NOT hallucinate.
"""
)

score_prompt = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="""
You are an AI recruiter.

Evaluate the candidate resume.

Job Description:
{job_description}

Resume:
{resume}

Give:
1. Match Score (0-100)
2. Explanation

SCORING RULES:
- More matching skills = higher score
- Missing required skills = lower score
- Relevant experience = better score

Return format:

Score:
Explanation:
"""
)

# =====================================================
# OUTPUT PARSER
# =====================================================

parser = StrOutputParser()

# =====================================================
# CHAINS
# =====================================================

extract_chain = extract_prompt | llm | parser

match_chain = match_prompt | llm | parser

score_chain = score_prompt | llm | parser

# =====================================================
# PIPELINE FUNCTION
# =====================================================

def run_pipeline(resume, candidate_name):

    print("\n" + "=" * 60)
    print(f"PROCESSING: {candidate_name}")
    print("=" * 60)

    # STEP 1
    print("\nSTEP 1: SKILL EXTRACTION\n")

    extraction = extract_chain.invoke({
        "resume": resume
    })

    print(extraction)

    # STEP 2
    print("\nSTEP 2: MATCHING ANALYSIS\n")

    matching = match_chain.invoke({
        "resume": resume,
        "job_description": job_description
    })

    print(matching)

    # STEP 3
    print("\nSTEP 3: FINAL SCORE\n")

    scoring = score_chain.invoke({
        "resume": resume,
        "job_description": job_description
    })

    print(scoring)

# =====================================================
# RUN ALL CANDIDATES
# =====================================================

run_pipeline(
    strong_resume,
    "STRONG CANDIDATE"
)

run_pipeline(
    average_resume,
    "AVERAGE CANDIDATE"
)

run_pipeline(
    weak_resume,
    "WEAK CANDIDATE"
)

# =====================================================
# DEBUG TEST
# =====================================================

bug_resume = """
Skills:
Python

Experience:
10 years as Chef
"""

run_pipeline(
    bug_resume,
    "DEBUG TEST"
)