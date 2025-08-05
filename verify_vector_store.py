#!/usr/bin/env python3
"""
Main script to run verification agent for spiritual document vector store.
This script uses CrewAI to orchestrate comprehensive verification of our vector database.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from utils.verification_utils import (
    verify_vector_store_integrity,
    test_similarity_search, 
    validate_metadata_preservation,
    check_embedding_quality,
    test_document_retrieval
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_agent_config(agent_file: str) -> dict:
    """Load agent configuration from YAML file."""
    import yaml
    try:
        with open(agent_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load agent config from {agent_file}: {str(e)}")
        raise

def load_task_config(task_file: str) -> dict:
    """Load task configuration from YAML file."""
    import yaml
    try:
        with open(task_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load task config from {task_file}: {str(e)}")
        raise

def create_verification_agent() -> Agent:
    """Create the verification agent with all necessary tools."""
    try:
        # Load agent configuration
        agent_config = load_agent_config("agents/verification_agent.yaml")
        config = agent_config["verification_agent"]
        
        # Create agent with tools
        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=config.get("verbose", True),
            memory=config.get("memory", True),
            tools=[
                verify_vector_store_integrity,
                test_similarity_search,
                validate_metadata_preservation,
                check_embedding_quality,
                test_document_retrieval
            ]
        )
        
        logger.info("Verification agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create verification agent: {str(e)}")
        raise

def create_verification_tasks(agent: Agent) -> list:
    """Create all verification tasks for the agent."""
    try:
        # Load task configurations
        task_config = load_task_config("tasks/verification_tasks.yaml")
        
        tasks = []
        
        # Create integrity verification task
        integrity_task = Task(
            description=task_config["verify_vector_store_integrity"]["description"],
            expected_output=task_config["verify_vector_store_integrity"]["expected_output"],
            agent=agent,
            tools=[verify_vector_store_integrity]
        )
        tasks.append(integrity_task)
        
        # Create similarity search task
        similarity_task = Task(
            description=task_config["test_similarity_search"]["description"],
            expected_output=task_config["test_similarity_search"]["expected_output"],
            agent=agent,
            tools=[test_similarity_search]
        )
        tasks.append(similarity_task)
        
        # Create metadata validation task
        metadata_task = Task(
            description=task_config["validate_metadata_preservation"]["description"],
            expected_output=task_config["validate_metadata_preservation"]["expected_output"],
            agent=agent,
            tools=[validate_metadata_preservation]
        )
        tasks.append(metadata_task)
        
        # Create embedding quality task
        embedding_task = Task(
            description=task_config["check_embedding_quality"]["description"],
            expected_output=task_config["check_embedding_quality"]["expected_output"],
            agent=agent,
            tools=[check_embedding_quality]
        )
        tasks.append(embedding_task)
        
        # Create document retrieval task
        retrieval_task = Task(
            description=task_config["test_document_retrieval"]["description"],
            expected_output=task_config["test_document_retrieval"]["expected_output"],
            agent=agent,
            tools=[test_document_retrieval]
        )
        tasks.append(retrieval_task)
        
        logger.info(f"Created {len(tasks)} verification tasks")
        return tasks
        
    except Exception as e:
        logger.error(f"Failed to create verification tasks: {str(e)}")
        raise

def run_verification_crew():
    """Run the verification crew to test the vector store."""
    try:
        logger.info("Starting vector store verification process...")
        
        # Check if vector store exists
        vector_store_path = os.getenv("VECTOR_STORE_DIR", "./vector_store")
        if not os.path.exists(vector_store_path):
            logger.error(f"Vector store not found at: {vector_store_path}")
            return {"status": "error", "error": "Vector store not found"}
        
        # Create verification agent
        verification_agent = create_verification_agent()
        
        # Create verification tasks
        verification_tasks = create_verification_tasks(verification_agent)
        
        # Create and run crew
        crew = Crew(
            agents=[verification_agent],
            tasks=verification_tasks,
            process=Process.sequential,
            verbose=True
        )
        
        logger.info("Running verification crew...")
        result = crew.kickoff()
        
        logger.info("Vector store verification completed successfully!")
        return {"status": "success", "result": result}
        
    except Exception as e:
        error_msg = f"Verification crew failed: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "error": error_msg}

def save_verification_report(results: dict, output_file: str = "verification_report.json"):
    """Save verification results to a report file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Verification report saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save verification report: {str(e)}")

def main():
    """Main function to run vector store verification."""
    try:
        print("=" * 60)
        print("SPIRITUAL DOCUMENT VECTOR STORE VERIFICATION")
        print("=" * 60)
        
        # Run verification
        results = run_verification_crew()
        
        # Save report
        save_verification_report(results)
        
        # Print summary
        print("\nVERIFICATION SUMMARY:")
        print("=" * 30)
        if results["status"] == "success":
            print("✅ Vector store verification completed successfully!")
            print(f"📊 Results: {results['result']}")
        else:
            print("❌ Vector store verification failed!")
            print(f"🚨 Error: {results['error']}")
        
        print("\n" + "=" * 60)
        return results
        
    except Exception as e:
        error_msg = f"Main verification process failed: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return {"status": "error", "error": error_msg}

if __name__ == "__main__":
    main()
