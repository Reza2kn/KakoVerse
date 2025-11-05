generate:
	uv run python Scripts/generate_persona_plan.py
	uv run python Scripts/generate_crisis_category_plan.py
	uv run python Scripts/generate_crisis_profiles.py
	uv run python Scripts/generate_persona_cards.py
	uv run python Scripts/generate_conversations.py

analyze:
	uv run python Scripts/analyze_conversations.py

clean:
	rm -rf artifacts outputs logs
