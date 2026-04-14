#!/usr/bin/env python3
"""
Build APEX-style Calibration Dataset

Reconstructs a calibration dataset matching the composition from
APEX Technical Report Appendix B using publicly available sources.

Distribution:
  - Multi-turn Chat:  30% (~15,000 tokens)
  - Code:             25% (~12,500 tokens)
  - Reasoning:        25% (~12,500 tokens)
  - Tool-calling:     20% (~10,000 tokens)

Total: ~50,000 tokens

Sources (all public, no Wikipedia):
  - Chat:      OpenOrca / ShareGPT-style conversations
  - Code:      CodeSearchNet samples
  - Reasoning: MathQA / step-by-step problem solving
  - Tool-use:  Function calling / API interaction patterns

Usage:
  pip install datasets tokenizers
  python build_calibration_data.py
"""

import json
import hashlib
import random
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Tuple

# Try to import tokenizers; fall back to approximate counting
try:
    from tokenizers import Tokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("Warning: 'tokenizers' not installed. Using approximate token count (words * 1.3).")
    print("Install with: pip install tokenizers")

try:
    from datasets import load_dataset
    HAS_DATETS = True
except ImportError:
    HAS_DATETS = False
    print("Error: 'datasets' library required. Install with: pip install datasets")
    import sys
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

TARGET_TOKENS = 50_000
DISTRIBUTION = OrderedDict([
    ("chat",       0.30),
    ("code",       0.25),
    ("reasoning",  0.25),
    ("tool_call",  0.20),
])

OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "calibration_data.jsonl"
OUTPUT_STATS = OUTPUT_DIR / "calibration_stats.json"

SEED = 42
random.seed(SEED)


# ──────────────────────────────────────────────────────────────
# Token counting
# ──────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if HAS_TOKENIZER:
        # Use a simple GPT-2 style tokenizer for estimation
        from tokenizers import Tokenizer as TK
        try:
            tok = TK.from_pretrained("gpt2")
            return len(tok.encode(text).tokens)
        except Exception:
            pass
    # Fallback: ~1 token ≈ 0.75 words
    return max(1, int(len(text.split()) * 1.3))


# ──────────────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────────────

def text_hash(text: str) -> str:
    """Create a hash for deduplication."""
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate(samples: List[Dict]) -> List[Dict]:
    """Remove exact duplicate entries based on content hash."""
    seen = set()
    unique = []
    for sample in samples:
        h = text_hash(sample["text"])
        if h not in seen:
            seen.add(h)
            unique.append(sample)
    return unique


# ──────────────────────────────────────────────────────────────
# Data collection functions
# ──────────────────────────────────────────────────────────────

def collect_chat_data(target_tokens: int) -> List[Dict]:
    """
    Collect multi-turn chat conversations.
    Source: OpenOrca / OpenAssistant style datasets.
    """
    print("  Collecting chat data...")
    samples = []

    try:
        # OpenAssistant conversations (public, multilingual)
        ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
        for item in ds:
            if item.get("text") and len(item["text"].strip()) > 50:
                samples.append({
                    "text": item["text"].strip(),
                    "source": "oasst1",
                    "domain": "chat",
                    "lang": "en" if all(ord(c) < 128 for c in item["text"][:200]) else "es"
                })
            if len(samples) >= 200:
                break
    except Exception as e:
        print(f"    Warning: Could not load OpenAssistant: {e}")

    # Fallback: generate synthetic chat if dataset unavailable
    if not samples:
        print("    Using synthetic chat data as fallback...")
        chat_templates = [
            {
                "text": "User: Can you explain how quantum computing works in simple terms?\nAssistant: Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously due to superposition. Unlike classical bits that are either 0 or 1, qubits can be both at once. This allows quantum computers to process many possibilities in parallel, making them powerful for specific problems like factoring large numbers or simulating molecules.",
                "source": "synthetic",
                "domain": "chat",
                "lang": "en"
            },
            {
                "text": "Usuario: ¿Cuáles son las mejores prácticas para aprender programación?\nAsistente: Las mejores prácticas incluyen: 1) Practicar diariamente con ejercicios, 2) Construir proyectos personales, 3) Leer código de otros desarrolladores, 4) Participar en comunidades como Stack Overflow, 5) Aprender los fundamentos antes de frameworks específicos.",
                "source": "synthetic",
                "domain": "chat",
                "lang": "es"
            },
            {
                "text": "User: What's the difference between supervised and unsupervised learning?\nAssistant: Supervised learning uses labeled training data where the correct answer is provided for each example. The model learns to map inputs to known outputs. Unsupervised learning works with unlabeled data, finding hidden patterns or groupings without knowing the 'right answer' beforehand. Examples include clustering and dimensionality reduction.",
                "source": "synthetic",
                "domain": "chat",
                "lang": "en"
            },
            {
                "text": "User: How do I implement authentication in a REST API?\nAssistant: There are several approaches to API authentication: 1) API Keys - simple tokens passed in headers, good for service-to-service communication. 2) JWT (JSON Web Tokens) - self-contained tokens that carry user identity and permissions, ideal for stateless authentication. 3) OAuth 2.0 - delegated authorization flow, commonly used for third-party access. 4) Session-based authentication - server-side sessions with cookies. Choose based on your security requirements and architecture.",
                "source": "synthetic",
                "domain": "chat",
                "lang": "en"
            },
            {
                "text": "User: What are microservices and when should I use them?\nAssistant: Microservices is an architectural style where an application is composed of small, independent services that communicate over a network. Each service owns a specific business capability.\n\nUse microservices when:\n- Your team is large and needs to work independently\n- You need different technology stacks for different parts\n- You require independent scaling of components\n- You want faster deployment cycles\n\nAvoid them when:\n- Your team is small\n- Your application is simple\n- You need strong consistency across components\n- Network latency is a critical concern",
                "source": "synthetic",
                "domain": "chat",
                "lang": "en"
            },
            {
                "text": "Usuario: ¿Puedes explicarme qué es Docker y para qué sirve?\nAsistente: Docker es una plataforma de contenerización que permite empaquetar aplicaciones con todas sus dependencias en contenedores. Los contenedores son ligeros, portables y se ejecutan de manera consistente en cualquier entorno.\n\nVentajas principales:\n1) Aislamiento: cada aplicación corre en su propio entorno\n2) Portabilidad: funciona igual en desarrollo, pruebas y producción\n3) Eficiencia: comparte el kernel del sistema operativo\n4) Versionado: puedes versionar tus contenedores como imágenes\n\nEs especialmente útil para microservicios y CI/CD pipelines.",
                "source": "synthetic",
                "domain": "chat",
                "lang": "es"
            },
            {
                "text": "User: Can you explain the concept of a neural network?\nAssistant: A neural network is a computing system inspired by biological brains. It consists of layers of interconnected nodes (neurons):\n\n1) Input layer - receives the raw data\n2) Hidden layers - process the data through weighted connections\n3) Output layer - produces the final result\n\nEach connection has a weight that adjusts during training. The network learns by comparing its output to the expected result and adjusting weights through backpropagation.\n\nDeep learning refers to networks with many hidden layers, capable of learning complex patterns in data like images, text, or audio.",
                "source": "synthetic",
                "domain": "chat",
                "lang": "en"
            },
            {
                "text": "User: What's the difference between SQL and NoSQL databases?\nAssistant: SQL databases (PostgreSQL, MySQL) use structured tables with fixed schemas and support complex queries with JOINs. They follow ACID properties for reliable transactions.\n\nNoSQL databases come in several types:\n- Document (MongoDB): JSON-like documents, flexible schema\n- Key-Value (Redis): simple key-value pairs, very fast\n- Column-family (Cassandra): optimized for large-scale data\n- Graph (Neo4j): relationships between entities\n\nChoose SQL when you need strong consistency and complex queries. Choose NoSQL for flexible schemas, horizontal scaling, or specific data models.",
                "source": "synthetic",
                "domain": "chat",
                "lang": "en"
            },
        ]
        # Generate varied copies by adding intro/outro variations
        intros = [
            "Let me help you with that.\n\n",
            "Here's my detailed answer:\n\n",
            "Great question! Let me explain.\n\n",
        ]
        otros = [
            "\n\nIs there anything else you'd like to know?",
            "\n\nLet me know if you have follow-up questions!",
            "\n\nFeel free to ask if anything is unclear.",
        ]
        
        num_needed = max(1, target_tokens // 100)
        for i in range(min(num_needed, 50)):
            base = chat_templates[i % len(chat_templates)]
            intro = intros[i % len(intros)]
            outro = otros[i % len(otros)]
            samples.append({
                "text": intro + base["text"] + outro,
                "source": base["source"],
                "domain": base["domain"],
                "lang": base["lang"],
            })

    return samples


def collect_code_data(target_tokens: int) -> List[Dict]:
    """
    Collect code examples with comments and documentation.
    Sources: CodeSearchNet, The Stack (filtered).
    """
    print("  Collecting code data...")
    samples = []

    try:
        # CodeSearchNet - Python
        for lang in ["python", "javascript"]:
            try:
                ds = load_dataset("code_search_net", lang, split="train", streaming=True)
                for item in ds:
                    if item.get("func_code_string") and item.get("func_documentation_string"):
                        code = item["func_code_string"]
                        doc = item["func_documentation_string"]
                        if code and len(code.strip()) > 30:
                            combined = f"# Documentation: {doc}\n{code}" if doc else code
                            samples.append({
                                "text": combined.strip(),
                                "source": f"code_search_net_{lang}",
                                "domain": "code",
                                "lang": lang
                            })
                    if len(samples) >= 150:
                        break
            except Exception as e:
                print(f"    Warning: Could not load CodeSearchNet ({lang}): {e}")
    except Exception:
        pass

    # Fallback: synthetic code examples
    if not samples:
        print("    Using synthetic code data as fallback...")
        code_examples = [
            {
                "text": '''def binary_search(arr: List[int], target: int) -> int:
    """
    Perform binary search on a sorted array.
    Returns the index of target if found, else -1.
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1''',
                "source": "synthetic",
                "domain": "code",
                "lang": "python"
            },
            {
                "text": '''/**
 * Debounce function to limit execution rate
 * @param {Function} func - Function to debounce
 * @param {number} wait - Delay in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait = 300) {
    let timeoutId = null;
    
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, wait);
    };
}''',
                "source": "synthetic",
                "domain": "code",
                "lang": "javascript"
            },
            {
                "text": '''class LinkedList:
    """A singly linked list implementation with common operations."""
    
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
    
    def __init__(self):
        self.head = None
        self._size = 0
    
    def append(self, data):
        """Add a node at the end of the list. O(n) time complexity."""
        new_node = self.Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1
    
    def prepend(self, data):
        """Add a node at the beginning of the list. O(1) time complexity."""
        new_node = self.Node(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1
    
    def delete(self, data):
        """Delete the first occurrence of data. O(n) time complexity."""
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return
            current = current.next
    
    def __len__(self):
        return self._size''',
                "source": "synthetic",
                "domain": "code",
                "lang": "python"
            },
            {
                "text": '''// Express.js middleware for request rate limiting
// Uses a sliding window algorithm to track requests per IP

const rateLimit = (options = {}) => {
    const {
        windowMs = 15 * 60 * 1000, // 15 minutes
        max = 100,                  // max requests per window
        message = 'Too many requests',
        statusCode = 429
    } = options;
    
    // Store for request counts: { ip: { count, resetTime } }
    const store = new Map();
    
    return (req, res, next) => {
        const ip = req.ip || req.connection.remoteAddress;
        const now = Date.now();
        
        // Initialize or reset window for new IP
        if (!store.has(ip) || now > store.get(ip).resetTime) {
            store.set(ip, {
                count: 0,
                resetTime: now + windowMs
            });
        }
        
        const record = store.get(ip);
        record.count++;
        
        // Set rate limit headers
        res.setHeader('X-RateLimit-Limit', max);
        res.setHeader('X-RateLimit-Remaining', Math.max(0, max - record.count));
        res.setHeader('X-RateLimit-Reset', record.resetTime);
        
        if (record.count > max) {
            return res.status(statusCode).json({ error: message });
        }
        
        next();
    };
};

module.exports = rateLimit;''',
                "source": "synthetic",
                "domain": "code",
                "lang": "javascript"
            },
            {
                "text": '''// System programming example: File I/O in C with error handling
// Demonstrates safe file reading with proper resource cleanup

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define BUFFER_SIZE 4096

/**
 * Read entire file contents into a dynamically allocated buffer.
 * 
 * @param filepath  Path to the file to read
 * @param out_size  Output parameter for file size (can be NULL)
 * @return          Dynamically allocated buffer with file contents, or NULL on error
 * 
 * Caller is responsible for freeing the returned buffer.
 * Returns NULL and sets errno on failure.
 */
char* read_file_contents(const char* filepath, size_t* out_size) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s': %s\\n", filepath, strerror(errno));
        return NULL;
    }
    
    // Get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    if (file_size < 0) {
        fprintf(stderr, "Error determining file size\\n");
        fclose(fp);
        errno = EIO;
        return NULL;
    }
    
    // Allocate buffer (null-terminated)
    char* buffer = malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Error allocating %ld bytes\\n", file_size + 1);
        fclose(fp);
        errno = ENOMEM;
        return NULL;
    }
    
    // Read file contents
    size_t bytes_read = fread(buffer, 1, file_size, fp);
    buffer[bytes_read] = '\\0';  // Null terminate
    
    if (bytes_read != (size_t)file_size && ferror(fp)) {
        fprintf(stderr, "Error reading file: %s\\n", strerror(errno));
        free(buffer);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    
    if (out_size) {
        *out_size = bytes_read;
    }
    
    return buffer;
}''',
                "source": "synthetic",
                "domain": "code",
                "lang": "c"
            },
            {
                "text": '''# Rust: Thread-safe concurrent cache with TTL support
# Demonstrates Arc, Mutex, and HashMap for safe concurrent access

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A thread-safe cache with time-to-live (TTL) for each entry.
/// 
/// # Example
/// ```
/// let cache = Cache::new(Duration::from_secs(60));
/// cache.insert("key", "value");
/// assert_eq!(cache.get("key"), Some("value"));
/// ```
pub struct Cache<K, V> {
    entries: Arc<Mutex<HashMap<K, CacheEntry<V>>>>,
    ttl: Duration,
}

struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
}

impl<K: std::cmp::Eq + std::hash::Hash + Clone, V: Clone> Cache<K, V> {
    /// Create a new cache with the specified default TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
            ttl,
        }
    }

    /// Insert a key-value pair into the cache.
    pub fn insert(&self, key: K, value: V) {
        let mut entries = self.entries.lock().unwrap();
        entries.insert(key, CacheEntry {
            value,
            inserted_at: Instant::now(),
        });
    }

    /// Get a value from the cache if it exists and hasn't expired.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut entries = self.entries.lock().unwrap();
        
        entries.get(key).and_then(|entry| {
            if entry.inserted_at.elapsed() > self.ttl {
                // Entry expired, remove it
                entries.remove(key);
                None
            } else {
                Some(entry.value.clone())
            }
        })
    }

    /// Remove expired entries from the cache.
    pub fn cleanup(&self) -> usize {
        let mut entries = self.entries.lock().unwrap();
        let initial_len = entries.len();
        
        entries.retain(|_, entry| entry.inserted_at.elapsed() <= self.ttl);
        
        initial_len - entries.len()
    }
}''',
                "source": "synthetic",
                "domain": "code",
                "lang": "rust"
            },
        ]
        # Generate varied copies by adding comment variations
        comment_prefixes = [
            "# Updated implementation with better error handling.\n",
            "# Refactored for clarity and performance.\n",
            "# Production-ready version with comprehensive docs.\n",
            "// Revised: added input validation and edge case handling.\n",
            "// Clean implementation following best practices.\n",
            "/* Version 2.0: Improved with community feedback. */\n",
        ]
        
        num_needed = max(1, target_tokens // 100)
        for i in range(min(num_needed, 50)):
            base = code_examples[i % len(code_examples)]
            prefix = comment_prefixes[i % len(comment_prefixes)]
            samples.append({
                "text": prefix + base["text"],
                "source": base["source"],
                "domain": base["domain"],
                "lang": base["lang"],
            })

    return samples


def collect_reasoning_data(target_tokens: int) -> List[Dict]:
    """
    Collect step-by-step reasoning and math problem solving.
    Sources: MathQA, PRM800K, or synthetic.
    """
    print("  Collecting reasoning data...")
    samples = []

    try:
        # Try to load a reasoning/math dataset
        ds = load_dataset("lighteval/MATH", split="train", streaming=True)
        for item in ds:
            if item.get("solution") or item.get("problem"):
                problem = item.get("problem", "")
                solution = item.get("solution", "")
                if len(problem) > 20 and len(solution) > 50:
                    samples.append({
                        "text": f"Problem: {problem}\n\nSolution: {solution}",
                        "source": "MATH",
                        "domain": "reasoning",
                        "lang": "en"
                    })
            if len(samples) >= 150:
                break
    except Exception as e:
        print(f"    Warning: Could not load MATH dataset: {e}")

    # Fallback: synthetic reasoning
    if not samples:
        print("    Using synthetic reasoning data as fallback...")
        reasoning_examples = [
            {
                "text": """Problem: A train travels 120 km in 2 hours. What is its average speed in m/s?

Solution: Let's solve this step by step.

Step 1: Calculate speed in km/h
Speed = Distance / Time = 120 km / 2 h = 60 km/h

Step 2: Convert km/h to m/s
We know: 1 km = 1000 m and 1 h = 3600 s
So: 60 km/h = 60 × 1000 m / 3600 s = 60000 / 3600 = 16.67 m/s

Answer: The train's average speed is 16.67 m/s.""",
                "source": "synthetic",
                "domain": "reasoning",
                "lang": "en"
            },
            {
                "text": """Problem: Prove that the sum of two even numbers is always even.

Solution: Let's prove this step by step.

Step 1: Define what makes a number even
An even number can be written as 2k where k is an integer.

Step 2: Let our two even numbers be 2a and 2b where a,b are integers.

Step 3: Add them
Sum = 2a + 2b = 2(a + b)

Step 4: Since a and b are integers, (a + b) is also an integer.
Therefore, 2(a + b) is divisible by 2, making it even.

Conclusion: The sum of two even numbers is always even. ∎""",
                "source": "synthetic",
                "domain": "reasoning",
                "lang": "en"
            },
            {
                "text": """Problem: If f(x) = 3x² + 2x - 5, find f'(x) and evaluate f'(2).

Solution: Let's find the derivative step by step.

Step 1: Apply the power rule to each term
The power rule states: d/dx(xⁿ) = nxⁿ⁻¹

For 3x²: derivative = 3 × 2x²⁻¹ = 6x
For 2x: derivative = 2 × 1x¹⁻¹ = 2
For -5: derivative = 0 (constant)

Step 2: Combine: f'(x) = 6x + 2

Step 3: Evaluate at x = 2
f'(2) = 6(2) + 2 = 12 + 2 = 14

Answer: f'(x) = 6x + 2 and f'(2) = 14.""",
                "source": "synthetic",
                "domain": "reasoning",
                "lang": "en"
            },
            {
                "text": """Problem: A rectangle has a perimeter of 40cm and an area of 96cm². Find its dimensions.

Solution: Let's solve this systematically.

Step 1: Define variables
Let length = l and width = w

Step 2: Set up equations from given information
Perimeter: 2l + 2w = 40, so l + w = 20
Area: l × w = 96

Step 3: Express l in terms of w
From l + w = 20: l = 20 - w

Step 4: Substitute into area equation
(20 - w) × w = 96
20w - w² = 96
w² - 20w + 96 = 0

Step 5: Solve the quadratic equation
Using the quadratic formula: w = (20 ± √(400 - 384)) / 2
w = (20 ± √16) / 2 = (20 ± 4) / 2
w = 12 or w = 8

If w = 12, then l = 8. If w = 8, then l = 12.

Answer: The rectangle is 12cm × 8cm.""",
                "source": "synthetic",
                "domain": "reasoning",
                "lang": "en"
            },
            {
                "text": """Problem: How many ways can you arrange the letters in the word "MATRIX"?

Solution: Let's work through this combinatorics problem.

Step 1: Count the total letters
MATRIX has 6 letters: M, A, T, R, I, X

Step 2: Check for repeated letters
All letters are unique (no repetitions)

Step 3: Apply the permutation formula
For n distinct objects, the number of arrangements is n!
Here n = 6, so arrangements = 6! = 6 × 5 × 4 × 3 × 2 × 1 = 720

Answer: There are 720 different arrangements of the letters in "MATRIX".""",
                "source": "synthetic",
                "domain": "reasoning",
                "lang": "en"
            },
            {
                "text": """Problem: Solve the system of equations:
2x + 3y = 12
4x - y = 5

Solution: Let's solve using substitution or elimination.

Step 1: Multiply the second equation by 3 to align y terms
4x - y = 5 → 12x - 3y = 15

Step 2: Add to the first equation
(2x + 3y) + (12x - 3y) = 12 + 15
14x = 27
x = 27/14 ≈ 1.93

Step 3: Substitute back to find y
4(27/14) - y = 5
108/14 - y = 5
y = 108/14 - 70/14 = 38/14 = 19/7 ≈ 2.71

Answer: x = 27/14, y = 19/7""",
                "source": "synthetic",
                "domain": "reasoning",
                "lang": "en"
            },
            {
                "text": """Problem: A fair die is rolled twice. What is the probability that the sum is 7?

Solution: Let's calculate this probability step by step.

Step 1: Identify the sample space
Each roll has 6 outcomes, so total outcomes = 6 × 6 = 36

Step 2: Count favorable outcomes (sum = 7)
Pairs that sum to 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)
That's 6 favorable outcomes

Step 3: Calculate probability
P(sum = 7) = favorable / total = 6/36 = 1/6 ≈ 0.167

Answer: The probability is 1/6 or approximately 16.7%.""",
                "source": "synthetic",
                "domain": "reasoning",
                "lang": "en"
            },
            {
                "text": """Problem: Find the next number in the sequence: 2, 6, 12, 20, 30, ?

Solution: Let's analyze the pattern step by step.

Step 1: Look at the differences between consecutive terms
6 - 2 = 4
12 - 6 = 6
20 - 12 = 8
30 - 20 = 10

Step 2: Notice the pattern in differences
The differences are: 4, 6, 8, 10 (increasing by 2 each time)

Step 3: Continue the pattern
Next difference = 10 + 2 = 12

Step 4: Find the next term
30 + 12 = 42

Alternative approach: The nth term follows the formula n(n+1)
Term 1: 1×2 = 2
Term 2: 2×3 = 6
Term 3: 3×4 = 12
Term 4: 4×5 = 20
Term 5: 5×6 = 30
Term 6: 6×7 = 42

Answer: The next number is 42.""",
                "source": "synthetic",
                "domain": "reasoning",
                "lang": "en"
            },
        ]
        # Generate varied copies by adding intro/outro variations
        intros = [
            "Let me help you with that.\n\n",
            "Here's my detailed answer:\n\n",
            "Great question! Let me explain.\n\n",
            "I'll break this down for you.\n\n",
            "Let's dive into this topic.\n\n",
            "Here's a comprehensive answer:\n\n",
        ]
        outros = [
            "\n\nIs there anything else you'd like to know?",
            "\n\nLet me know if you have follow-up questions!",
            "\n\nFeel free to ask if anything is unclear.",
            "\n\nI hope this helps! What else can I assist you with?",
            "\n\nWould you like me to elaborate on any part?",
        ]
        
        num_needed = max(1, target_tokens // 100)
        for i in range(min(num_needed, 80)):
            base = reasoning_examples[i % len(reasoning_examples)]
            intro = intros[i % len(intros)]
            outro = outros[i % len(outros)]
            samples.append({
                "text": intro + base["text"] + outro,
                "source": base["source"],
                "domain": base["domain"],
                "lang": base["lang"],
            })

    return samples


def collect_tool_call_data(target_tokens: int) -> List[Dict]:
    """
    Collect tool-calling and function-calling examples.
    Sources: Berkeley Function Calling, NexusRaven.
    """
    print("  Collecting tool-call data...")
    samples = []

    try:
        # Try Berkeley function calling dataset
        ds = load_dataset("Berkeley-NLP/ToolBench", split="train", streaming=True)
        for item in ds:
            if item.get("text"):
                samples.append({
                    "text": item["text"].strip(),
                    "source": "toolbench",
                    "domain": "tool_call",
                    "lang": "en"
                })
            if len(samples) >= 120:
                break
    except Exception as e:
        print(f"    Warning: Could not load ToolBench: {e}")

    # Fallback: synthetic tool-calling examples
    if not samples:
        print("    Using synthetic tool-call data as fallback...")
        tool_examples = [
            {
                "text": '''User: What's the weather in San Francisco?

[Tool Call]
{
    "name": "get_weather",
    "arguments": {
        "location": "San Francisco, CA",
        "units": "celsius"
    }
}

[Tool Response]
{
    "temperature": 18,
    "conditions": "partly cloudy",
    "humidity": 72
}

Assistant: The current weather in San Francisco is 18°C with partly cloudy conditions and 72% humidity.''',
                "source": "synthetic",
                "domain": "tool_call",
                "lang": "en"
            },
            {
                "text": '''User: Search for flights from NYC to London next week.

[Tool Call]
{
    "name": "search_flights",
    "arguments": {
        "origin": "NYC",
        "destination": "LON",
        "departure_date": "2024-04-20",
        "passengers": 1,
        "class": "economy"
    }
}

[Tool Response]
[
    {"airline": "BA", "flight": "BA178", "price": 450, "duration": "7h 15m"},
    {"airline": "VS", "flight": "VS4", "price": 485, "duration": "7h 05m"}
]

Assistant: I found several flights. The cheapest is British Airways BA178 at $450 with a flight time of 7h 15m.''',
                "source": "synthetic",
                "domain": "tool_call",
                "lang": "en"
            },
            {
                "text": '''User: Convert 100 USD to EUR.

[Tool Call]
{
    "name": "convert_currency",
    "arguments": {
        "from_currency": "USD",
        "to_currency": "EUR",
        "amount": 100
    }
}

[Tool Response]
{
    "from_currency": "USD",
    "to_currency": "EUR",
    "amount": 100,
    "converted_amount": 92.50,
    "exchange_rate": 0.925,
    "timestamp": "2024-04-15T10:30:00Z"
}

Assistant: 100 USD equals 92.50 EUR at the current exchange rate of 0.925.''',
                "source": "synthetic",
                "domain": "tool_call",
                "lang": "en"
            },
            {
                "text": '''User: Set a reminder for tomorrow at 3pm to review the quarterly report.

[Tool Call]
{
    "name": "set_reminder",
    "arguments": {
        "title": "Review quarterly report",
        "datetime": "2024-04-16T15:00:00",
        "timezone": "America/New_York",
        "notification_type": "push"
    }
}

[Tool Response]
{
    "reminder_id": "rem_8472",
    "title": "Review quarterly report",
    "datetime": "2024-04-16T15:00:00-04:00",
    "status": "scheduled"
}

Assistant: I've set a reminder for tomorrow at 3:00 PM ET to review the quarterly report. You'll receive a push notification.''',
                "source": "synthetic",
                "domain": "tool_call",
                "lang": "en"
            },
            {
                "text": '''User: What's the status of my GitHub Actions workflow #423?

[Tool Call]
{
    "name": "github.get_workflow_run",
    "arguments": {
        "owner": "myorg",
        "repo": "myproject",
        "run_id": 423
    }
}

[Tool Response]
{
    "id": 423,
    "name": "CI Pipeline",
    "status": "completed",
    "conclusion": "success",
    "created_at": "2024-04-15T09:00:00Z",
    "updated_at": "2024-04-15T09:12:34Z",
    "jobs": [
        {"name": "build", "status": "completed", "conclusion": "success"},
        {"name": "test", "status": "completed", "conclusion": "success"},
        {"name": "lint", "status": "completed", "conclusion": "success"}
    ]
}

Assistant: Workflow #423 "CI Pipeline" completed successfully. All jobs (build, test, lint) passed. It ran for about 12 minutes.''',
                "source": "synthetic",
                "domain": "tool_call",
                "lang": "en"
            },
            {
                "text": '''User: Find restaurants near Central Park with ratings above 4.5.

[Tool Call]
{
    "name": "yelp.search",
    "arguments": {
        "location": "Central Park, NYC",
        "radius": 1000,
        "categories": "restaurants",
        "min_rating": 4.5,
        "limit": 5
    }
}

[Tool Response]
[
    {"name": "The River Café", "rating": 4.8, "price": "$$$$", "cuisine": "American"},
    {"name": "Daniel", "rating": 4.7, "price": "$$$$", "cuisine": "French"},
    {"name": "Per Se", "rating": 4.6, "price": "$$$$", "cuisine": "Contemporary"}
]

Assistant: I found 3 highly-rated restaurants near Central Park. The top choice is The River Café with a 4.8 rating, offering American cuisine.''',
                "source": "synthetic",
                "domain": "tool_call",
                "lang": "en"
            },
            {
                "text": '''User: Calculate the compound interest on $5000 at 6% annual rate for 3 years, compounded monthly.

[Tool Call]
{
    "name": "calculator.evaluate",
    "arguments": {
        "expression": "5000 * (1 + 0.06/12)^(12*3)"
    }
}

[Tool Response]
{
    "result": 5983.44,
    "expression": "5000 * (1 + 0.06/12)^(12*3)",
    "precision": 2
}

Assistant: The compound interest calculation: $5,000 at 6% annual rate for 3 years compounded monthly equals $5,983.44. That's $983.44 in interest earned.''',
                "source": "synthetic",
                "domain": "tool_call",
                "lang": "en"
            },
            {
                "text": '''User: Send an email to team@company.com with subject "Meeting Notes" and body "Please find the attached notes from today's standup."

[Tool Call]
{
    "name": "email.send",
    "arguments": {
        "to": "team@company.com",
        "subject": "Meeting Notes",
        "body": "Please find the attached notes from today's standup.",
        "attachments": []
    }
}

[Tool Response]
{
    "message_id": "msg_92847",
    "status": "sent",
    "timestamp": "2024-04-15T11:45:00Z"
}

Assistant: Email sent successfully to team@company.com with subject "Meeting Notes". Message ID: msg_92847.''',
                "source": "synthetic",
                "domain": "tool_call",
                "lang": "en"
            },
        ]
        # Generate varied copies by adding intro/outro variations
        intros = [
            "Let me help you with that.\n\n",
            "Here's my detailed answer:\n\n",
            "Great question! Let me explain.\n\n",
            "I'll walk you through this.\n\n",
            "Let's get started.\n\n",
            "Here's a comprehensive answer:\n\n",
        ]
        outros = [
            "\n\nIs there anything else you'd like to know?",
            "\n\nLet me know if you have follow-up questions!",
            "\n\nFeel free to ask if anything is unclear.",
            "\n\nI hope this helps! What else can I assist you with?",
            "\n\nWould you like me to elaborate on any part?",
        ]
        
        num_needed = max(1, target_tokens // 100)
        for i in range(min(num_needed, 60)):
            base = tool_examples[i % len(tool_examples)]
            intro = intros[i % len(intros)]
            outro = outros[i % len(outros)]
            samples.append({
                "text": intro + base["text"] + outro,
                "source": base["source"],
                "domain": base["domain"],
                "lang": base["lang"],
            })

    return samples


# ──────────────────────────────────────────────────────────────
# Main build pipeline
# ──────────────────────────────────────────────────────────────

def build_dataset():
    """Build the complete calibration dataset."""
    print("=" * 60)
    print("Building APEX-style Calibration Dataset")
    print("=" * 60)
    print(f"\nTarget: {TARGET_TOKENS:,} tokens")
    print(f"Distribution: {dict(DISTRIBUTION)}\n")

    # Collect data for each domain
    all_samples = {}
    for domain, ratio in DISTRIBUTION.items():
        target = int(TARGET_TOKENS * ratio)
        print(f"[{domain}] Target: ~{target:,} tokens ({ratio*100:.0f}%)")

        if domain == "chat":
            all_samples[domain] = collect_chat_data(target)
        elif domain == "code":
            all_samples[domain] = collect_code_data(target)
        elif domain == "reasoning":
            all_samples[domain] = collect_reasoning_data(target)
        elif domain == "tool_call":
            all_samples[domain] = collect_tool_call_data(target)

    # Deduplicate each domain
    print("\nDeduplicating...")
    stats = {}
    for domain in all_samples:
        before = len(all_samples[domain])
        all_samples[domain] = deduplicate(all_samples[domain])
        after = len(all_samples[domain])
        stats[domain] = {"before_dedup": before, "after_dedup": after}
        print(f"  {domain}: {before} -> {after} entries")

    # Count tokens and truncate to target
    print("\nCounting tokens and truncating...")
    final_samples = []
    for domain, ratio in DISTRIBUTION.items():
        target_tokens = int(TARGET_TOKENS * ratio)
        
        # Count tokens for all samples first
        for sample in all_samples[domain]:
            sample["token_count"] = count_tokens(sample["text"])
        
        # Shuffle to get a mix of lengths, then take samples up to target
        shuffled = list(all_samples[domain])
        random.shuffle(shuffled)
        
        domain_tokens = 0
        domain_samples = []
        for sample in shuffled:
            tokens = sample["token_count"]
            if domain_tokens + tokens > target_tokens and domain_samples:
                break
            domain_tokens += tokens
            domain_samples.append(sample)

        final_samples.extend(domain_samples)
        stats[domain]["tokens"] = domain_tokens
        stats[domain]["samples"] = len(domain_samples)
        print(f"  {domain}: {domain_tokens:,} tokens ({len(domain_samples)} samples)")

    # Shuffle final dataset
    random.shuffle(final_samples)

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Save stats
    total_tokens = sum(s["tokens"] for s in stats.values())
    total_samples = sum(s["samples"] for s in stats.values())
    stats["total"] = {"tokens": total_tokens, "samples": total_samples}

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Complete!")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Stats: {OUTPUT_STATS}")
    print(f"{'=' * 60}")

    return stats


if __name__ == "__main__":
    build_dataset()
