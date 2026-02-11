# LangChain LLM Chain Basics

Introduction to LangChain framework fundamentals, covering the basic concepts of Large Language Model (LLM) chains and prompt templates using OpenAI.

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Requirements for running the project:

- [Python 3.9+](https://www.python.org/)
- [OpenAI API Key](https://platform.openai.com/api-keys) - For LLM access
- [LangChain](https://python.langchain.com/) - LLM framework
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variables management

### Installing

A step by step series to get a development environment running:

1. Clone the repository

    ```bash
    git clone https://github.com/AnderssonProgramming/langchain-llm-chain-basics.git
    cd langchain-llm-chain-basics
    ```

2. Create and activate a virtual environment

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. Install the required libraries

    ```bash
    pip install langchain langchain-openai python-dotenv
    ```

4. Create a `.env` file with your OpenAI API key

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

5. Run the main script

    ```bash
    python main.py
    ```

## Introduction and Motivation

This project serves as a foundational introduction to the LangChain framework, which has become a standard tool for building applications powered by Large Language Models (LLMs). Understanding LangChain basics is essential before moving to more advanced implementations like Retrieval-Augmented Generation (RAG).

### What is LangChain?

LangChain is a framework designed to simplify the creation of applications using large language models. It provides:

- **Prompt Templates**: Standardized way to structure prompts
- **LLM Chains**: Sequential processing of prompts through models
- **Memory**: Context retention across interactions
- **Agents**: Autonomous decision-making capabilities

### Learning Objectives

By completing this tutorial, you will understand:

1. How to set up and configure LangChain with OpenAI
2. Creating and using prompt templates
3. Building basic LLM chains
4. Handling model responses and outputs

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Input                           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               Prompt Template                           │
│         (Structures the input for the LLM)              │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  LLM Chain                              │
│           (Processes through OpenAI)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               Model Response                            │
│            (Generated output)                           │
└─────────────────────────────────────────────────────────┘
```

## Repository Structure

```
/
├── README.md           # Project documentation
├── LICENSE             # MIT License
├── .env.example        # Environment variables template
├── requirements.txt    # Python dependencies
└── main.py             # Main application script
```

## Usage Example

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
response = chain.run("machine learning")
print(response)
```

## Built With

- [Python](https://www.python.org/) - Programming language
- [LangChain](https://python.langchain.com/) - LLM application framework
- [OpenAI](https://openai.com/) - Large Language Model provider

## References

- [LangChain Quickstart Tutorial](https://python.langchain.com/docs/get_started/quickstart)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## Authors

- **Andersson David Sánchez Méndez** - *Developer* - [AnderssonProgramming](https://github.com/AnderssonProgramming)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain LLM Chain Tutorial for foundational concepts
- OpenAI for providing the GPT models
- AREP Course - Introduction to RAGs Lab