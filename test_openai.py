from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()


def test_openai_connection():
    """Test OpenAI connection"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OpenAI API key not found in environment variables")
            return False

        print(f"✅ Found OpenAI API key: {api_key[:10]}...")

        # Initialize client
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")

        # Test a simple API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, World!' in JSON format."},
            ],
            max_tokens=50,
        )

        print("✅ API call successful")
        print(f"Response: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"❌ OpenAI connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing OpenAI connection...")
    success = test_openai_connection()
    if success:
        print(
            "\n🎉 OpenAI connection test passed! You can now run the main application."
        )
    else:
        print(
            "\n💥 OpenAI connection test failed. Please check your API key and network connection."
        )
