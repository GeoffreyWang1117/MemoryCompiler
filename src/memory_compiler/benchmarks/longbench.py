"""LongBench benchmark implementation.

LongBench is a multi-task benchmark for long context understanding
in large language models.

Reference: https://github.com/THUDM/LongBench
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

from memory_compiler.benchmarks.base import BenchmarkDataset, BenchmarkSample


class LongBenchBenchmark(BenchmarkDataset):
    """LongBench benchmark for long context understanding.

    Focuses on multi-turn dialogue summarization and QA tasks.

    Example:
        >>> benchmark = LongBenchBenchmark(task="multi_turn_qa")
        >>> benchmark.load()
        >>> print(len(benchmark))
    """

    # Available tasks relevant to dialogue memory
    DIALOGUE_TASKS = [
        "multi_turn_qa",
        "dialogue_summarization",
        "passage_retrieval",
    ]

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        task: str = "multi_turn_qa",
        split: str = "test",
    ) -> None:
        """Initialize LongBench benchmark.

        Args:
            data_dir: Directory containing LongBench data.
            task: Task type to evaluate.
            split: Data split.
        """
        super().__init__(data_dir)
        self.task = task
        self.split = split
        self._samples: List[BenchmarkSample] = []

        if task not in self.DIALOGUE_TASKS:
            logger.warning(
                f"Task '{task}' may not be dialogue-focused. "
                f"Recommended tasks: {self.DIALOGUE_TASKS}"
            )

    @property
    def name(self) -> str:
        """Get benchmark name."""
        return f"LongBench-{self.task}-{self.split}"

    def load(self) -> None:
        """Load LongBench dataset."""
        if self.data_dir and self.data_dir.exists():
            self._load_from_files()
        else:
            logger.warning("LongBench data not found, using synthetic samples")
            self._load_synthetic()

    def _load_from_files(self) -> None:
        """Load from actual LongBench data files."""
        data_file = self.data_dir / f"{self.task}.jsonl"

        if not data_file.exists():
            logger.warning(f"LongBench file not found: {data_file}")
            self._load_synthetic()
            return

        with open(data_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                sample = self._parse_item(item, idx)
                if sample:
                    self._samples.append(sample)

        logger.info(f"Loaded {len(self._samples)} LongBench samples")

    def _parse_item(self, item: Dict[str, Any], idx: int) -> Optional[BenchmarkSample]:
        """Parse an item from LongBench format."""
        context = item.get("context", item.get("input", ""))

        # Extract questions if available
        questions = []
        if "questions" in item:
            questions = item["questions"]
        elif "query" in item:
            questions = [{"question": item["query"], "answer": item.get("answer", "")}]

        # Try to extract key facts from answers
        key_facts = []
        for q in questions:
            if q.get("answer"):
                key_facts.append(q["answer"])

        return BenchmarkSample(
            id=f"longbench_{self.task}_{idx}",
            dialogue=context,
            reference_summary=item.get("summary"),
            key_entities=item.get("entities", []),
            key_facts=key_facts,
            questions=questions,
            metadata={
                "task": self.task,
                "length": len(context),
            },
        )

    def _load_synthetic(self) -> None:
        """Generate synthetic LongBench-like samples."""
        if self.task == "multi_turn_qa":
            self._generate_qa_samples()
        elif self.task == "dialogue_summarization":
            self._generate_summarization_samples()
        else:
            self._generate_qa_samples()  # Default to QA

    def _generate_qa_samples(self) -> None:
        """Generate multi-turn QA samples."""
        samples_data = [
            {
                "context": """
User: I need help planning my wedding. We're thinking June 15th at a vineyard.
Assistant: A vineyard wedding in June sounds beautiful! How many guests?
User: About 150 guests. My fiancé Sarah wants a rustic theme.
Assistant: Rustic vineyard with 150 guests is lovely. Have you set a budget?
User: We're aiming for around $40,000 total.
Assistant: That's workable for 150 guests. Venue usually takes 40-50%.
User: The venue quoted us $15,000 including tables and chairs.
Assistant: That's reasonable at about 37% of budget. Leaves room for catering.
User: Sarah's cousin is a photographer, offering a discount.
Assistant: Family connections help! What about catering preferences?
User: We want farm-to-table since it's a vineyard setting.
Assistant: Farm-to-table fits perfectly. Local vendors would complement that.
User: Any suggestions for the ceremony music?
Assistant: String quartet for ceremony, band for reception is popular.
User: Sarah plays violin, maybe she could do a piece.
Assistant: That would be such a personal touch! During cocktail hour perhaps?
""",
                "questions": [
                    {"question": "When is the wedding planned?", "answer": "June 15th"},
                    {"question": "How many guests are expected?", "answer": "150"},
                    {"question": "What is the wedding budget?", "answer": "$40,000"},
                    {"question": "What is the fiancé's name?", "answer": "Sarah"},
                    {"question": "What type of venue is being considered?", "answer": "vineyard"},
                ],
                "entities": ["Sarah", "vineyard", "wedding"],
            },
            {
                "context": """
User: I'm starting a small bakery business. Need advice on permits.
Assistant: Exciting venture! What type of baked goods will you sell?
User: Artisan breads and French pastries. I trained in Paris.
Assistant: Wonderful! You'll need a food handler's license and business permit.
User: I found a location on Main Street, 800 square feet.
Assistant: Good size for a bakery. Check zoning for food service.
User: The landlord mentioned it was previously a cafe.
Assistant: That helps! Existing food service setup may ease permits.
User: My business partner Tom is handling the finances.
Assistant: Great to have a partner. Is Tom investing capital too?
User: Yes, we're each putting in $50,000 for startup.
Assistant: $100,000 total is solid. Equipment will be a big expense.
User: I found used ovens from a closing bakery for $15,000.
Assistant: Used equipment savings help a lot. Get them inspected.
User: We're naming it "Petit Paris Bakery" after my training.
Assistant: Love the name! Very fitting for French pastries.
User: Opening target is March 1st, before the spring rush.
Assistant: Smart timing. That gives you a few months to prepare.
""",
                "questions": [
                    {"question": "What will the bakery specialize in?", "answer": "Artisan breads and French pastries"},
                    {"question": "Where is the location?", "answer": "Main Street"},
                    {"question": "Who is the business partner?", "answer": "Tom"},
                    {"question": "What is the total startup investment?", "answer": "$100,000"},
                    {"question": "What is the bakery's name?", "answer": "Petit Paris Bakery"},
                ],
                "entities": ["Tom", "Petit Paris Bakery", "Main Street", "Paris"],
            },
            {
                "context": """
User: I adopted a rescue dog last week. Her name is Luna.
Assistant: Congratulations on adopting Luna! What breed is she?
User: She's a 2-year-old husky mix. Very energetic.
Assistant: Huskies need lots of exercise! Do you have a yard?
User: Yes, but she keeps digging under the fence.
Assistant: Classic husky behavior! You might need to bury chicken wire.
User: I tried that. Now she's jumping over the 4-foot fence.
Assistant: Huskies are escape artists. Consider a 6-foot fence.
User: My neighbor Mr. Wilson complained about the barking too.
Assistant: Huskies are vocal. Training and exercise can help reduce it.
User: I've been running with her every morning, 3 miles.
Assistant: Great! That should help burn her energy. Any improvement?
User: She's calmer during the day now. Still barks at squirrels.
Assistant: Squirrel-chasing is instinctive. Redirect with toys.
User: The vet said she's healthy but slightly underweight at 45 pounds.
Assistant: Vet visits are important. Feed her high-quality food.
User: Dr. Chen recommended a salmon-based diet.
Assistant: Salmon is great for coat health too, especially for huskies.
""",
                "questions": [
                    {"question": "What is the dog's name?", "answer": "Luna"},
                    {"question": "What breed is the dog?", "answer": "husky mix"},
                    {"question": "How old is the dog?", "answer": "2 years old"},
                    {"question": "Who is the neighbor?", "answer": "Mr. Wilson"},
                    {"question": "What did the vet recommend?", "answer": "salmon-based diet"},
                ],
                "entities": ["Luna", "Mr. Wilson", "Dr. Chen", "husky"],
            },
        ]

        for idx, data in enumerate(samples_data):
            self._samples.append(
                BenchmarkSample(
                    id=f"longbench_qa_synthetic_{idx}",
                    dialogue=data["context"].strip(),
                    key_entities=data["entities"],
                    key_facts=[q["answer"] for q in data["questions"]],
                    questions=data["questions"],
                    metadata={"synthetic": True, "task": "multi_turn_qa"},
                )
            )

        logger.info(f"Generated {len(self._samples)} synthetic LongBench QA samples")

    def _generate_summarization_samples(self) -> None:
        """Generate dialogue summarization samples."""
        samples_data = [
            {
                "context": """
User: I'm concerned about my teenager's screen time.
Assistant: How many hours a day are we talking about?
User: At least 6-7 hours, mostly on social media.
Assistant: That is significant. Have you discussed limits with them?
User: We tried, but my son Jake gets very defensive.
Assistant: That's common at this age. What specifically concerns you?
User: His grades dropped from As to Cs this semester.
Assistant: Academic impact is serious. Any changes in social behavior?
User: He doesn't hang out with friends in person anymore.
Assistant: Social isolation with heavy screen use is a red flag.
User: His pediatrician Dr. Martinez suggested family therapy.
Assistant: That's a good recommendation. Have you looked into it?
User: We have an appointment next Tuesday with a counselor.
Assistant: Great first step. Be open about your concerns in the session.
""",
                "summary": "Parent is worried about 16-year-old son Jake's excessive screen time (6-7 hours daily on social media), which has led to declining grades (A to C) and social isolation. Following pediatrician Dr. Martinez's advice, they've scheduled family therapy for next Tuesday.",
                "entities": ["Jake", "Dr. Martinez"],
                "facts": [
                    "Screen time is 6-7 hours daily",
                    "Grades dropped from As to Cs",
                    "Son's name is Jake",
                    "Dr. Martinez recommended therapy",
                ],
            },
            {
                "context": """
User: I want to learn a new language before my trip to Italy.
Assistant: When is your trip planned?
User: In 6 months, we leave September 10th.
Assistant: Six months is good for basics. Have you studied Italian before?
User: I took Spanish in high school, so I know some Romance language basics.
Assistant: That helps! Spanish and Italian share similarities.
User: What's the best app for learning?
Assistant: Duolingo is popular, but Babbel is more conversation-focused.
User: I tried Duolingo but found it too gamified.
Assistant: Then try Babbel or italki for tutor sessions.
User: My wife Maria already speaks Italian fluently.
Assistant: Perfect! Practice with her daily for best results.
User: She's from Florence originally.
Assistant: Florentine Italian is beautiful. She can teach you proper pronunciation.
User: We're visiting her family there for two weeks.
Assistant: Immersion will accelerate your learning dramatically.
""",
                "summary": "User is learning Italian for a September 10th trip to Florence to visit wife Maria's family. Has Spanish background but found Duolingo too gamified; considering Babbel or italki. Will practice with Italian-speaking wife from Florence for two weeks of immersion.",
                "entities": ["Maria", "Florence", "Italy"],
                "facts": [
                    "Trip is on September 10th",
                    "Trip is to Italy/Florence",
                    "Wife Maria speaks Italian",
                    "Maria is from Florence",
                    "Studied Spanish before",
                ],
            },
        ]

        for idx, data in enumerate(samples_data):
            self._samples.append(
                BenchmarkSample(
                    id=f"longbench_summ_synthetic_{idx}",
                    dialogue=data["context"].strip(),
                    reference_summary=data["summary"],
                    key_entities=data["entities"],
                    key_facts=data["facts"],
                    metadata={"synthetic": True, "task": "dialogue_summarization"},
                )
            )

        logger.info(f"Generated {len(self._samples)} synthetic summarization samples")

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self._samples)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Iterate over samples."""
        return iter(self._samples)

    @classmethod
    def available_tasks(cls) -> List[str]:
        """Get list of available dialogue-focused tasks."""
        return cls.DIALOGUE_TASKS.copy()
