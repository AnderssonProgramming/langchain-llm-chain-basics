"""
LangChain LLM Chain Basics - Main Application

This script demonstrates the fundamentals of LangChain framework:
- Setting up OpenAI LLM connection
- Creating prompt templates
- Building and running LLM chains
- Handling model responses

Author: Andersson David Sánchez Méndez
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Please create a .env file with your API key."
        )


def create_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> ChatOpenAI:
    """
    Initialize the OpenAI LLM.
    
    Args:
        model: The OpenAI model to use.
        temperature: Controls randomness in responses (0-1).
    
    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(model=model, temperature=temperature)


def create_prompt_template(template: str, input_variables: list) -> PromptTemplate:
    """
    Create a prompt template for structuring inputs.
    
    Args:
        template: The template string with placeholders.
        input_variables: List of variable names in the template.
    
    Returns:
        Configured PromptTemplate instance.
    """
    return PromptTemplate(
        input_variables=input_variables,
        template=template
    )


def create_chain(llm: ChatOpenAI, prompt: PromptTemplate) -> LLMChain:
    """
    Create an LLM chain combining the model and prompt.
    
    Args:
        llm: The language model instance.
        prompt: The prompt template.
    
    Returns:
        Configured LLMChain instance.
    """
    return LLMChain(llm=llm, prompt=prompt)


def demonstrate_simple_chain() -> None:
    """Demonstrate a basic LLM chain with a simple explanation prompt."""
    print("\n" + "=" * 60)
    print("DEMO 1: Simple Explanation Chain")
    print("=" * 60)
    
    llm = create_llm()
    
    prompt = create_prompt_template(
        template="Explain {topic} in simple terms that a beginner could understand.",
        input_variables=["topic"]
    )
    
    chain = create_chain(llm, prompt)
    
    topics = ["machine learning", "neural networks", "natural language processing"]
    
    for topic in topics:
        print(f"\n📚 Topic: {topic}")
        print("-" * 40)
        response = chain.run(topic)
        print(response)


def demonstrate_creative_chain() -> None:
    """Demonstrate a creative writing chain with higher temperature."""
    print("\n" + "=" * 60)
    print("DEMO 2: Creative Writing Chain")
    print("=" * 60)
    
    llm = create_llm(temperature=0.9)
    
    prompt = create_prompt_template(
        template=(
            "Write a short creative story (2-3 paragraphs) about {subject} "
            "in the genre of {genre}."
        ),
        input_variables=["subject", "genre"]
    )
    
    chain = create_chain(llm, prompt)
    
    print("\n✍️ Generating creative story...")
    print("-" * 40)
    response = chain.run(subject="a robot learning to paint", genre="science fiction")
    print(response)


def demonstrate_structured_output_chain() -> None:
    """Demonstrate a chain that produces structured output."""
    print("\n" + "=" * 60)
    print("DEMO 3: Structured Output Chain")
    print("=" * 60)
    
    llm = create_llm(temperature=0.3)
    
    prompt = create_prompt_template(
        template=(
            "Analyze the following concept and provide a structured response:\n\n"
            "Concept: {concept}\n\n"
            "Please provide:\n"
            "1. Definition (1-2 sentences)\n"
            "2. Key characteristics (3 bullet points)\n"
            "3. Real-world applications (2-3 examples)\n"
            "4. Related concepts (2-3 items)"
        ),
        input_variables=["concept"]
    )
    
    chain = create_chain(llm, prompt)
    
    print("\n🔍 Analyzing concept: 'API (Application Programming Interface)'")
    print("-" * 40)
    response = chain.run(concept="API (Application Programming Interface)")
    print(response)


def demonstrate_translation_chain() -> None:
    """Demonstrate a translation chain."""
    print("\n" + "=" * 60)
    print("DEMO 4: Translation Chain")
    print("=" * 60)
    
    llm = create_llm(temperature=0.3)
    
    prompt = create_prompt_template(
        template=(
            "Translate the following text from {source_language} to {target_language}.\n\n"
            "Text: {text}\n\n"
            "Translation:"
        ),
        input_variables=["source_language", "target_language", "text"]
    )
    
    chain = create_chain(llm, prompt)
    
    print("\n🌐 Translating text...")
    print("-" * 40)
    response = chain.run(
        source_language="English",
        target_language="Spanish",
        text="Artificial intelligence is transforming how we interact with technology."
    )
    print(response)


def interactive_mode() -> None:
    """Run an interactive mode where users can ask questions."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Ask any question and get an AI-powered response.")
    print("Type 'exit' to quit.\n")
    
    llm = create_llm(temperature=0.7)
    
    prompt = create_prompt_template(
        template=(
            "You are a helpful AI assistant. Answer the following question "
            "clearly and concisely:\n\n{question}"
        ),
        input_variables=["question"]
    )
    
    chain = create_chain(llm, prompt)
    
    while True:
        question = input("\n🤔 Your question: ").strip()
        
        if question.lower() == "exit":
            print("\nGoodbye! 👋")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        print("\n💡 Response:")
        print("-" * 40)
        response = chain.run(question=question)
        print(response)


def main() -> None:
    """Main entry point for the application."""
    print("=" * 60)
    print("🔗 LangChain LLM Chain Basics")
    print("=" * 60)
    
    # Load environment variables
    load_environment()
    print("✅ Environment loaded successfully!")
    
    # Run demonstrations
    demonstrate_simple_chain()
    demonstrate_creative_chain()
    demonstrate_structured_output_chain()
    demonstrate_translation_chain()
    
    # Ask if user wants interactive mode
    print("\n" + "=" * 60)
    user_input = input("Would you like to enter interactive mode? (yes/no): ").strip().lower()
    
    if user_input in ["yes", "y"]:
        interactive_mode()
    else:
        print("\n✅ All demonstrations completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()
