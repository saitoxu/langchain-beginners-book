from langsmith import Client

run_id = "562e56e1-9feb-46e1-9ed9-ee2c3f879a75"

client = Client()
feedback = client.create_feedback(
    run_id=run_id,
    key="thumbs",
    score=0,
)
print(feedback)
