"""
Universal HayStack Pipeline - Canonical Schema Version
Updated to use canonical schema views instead of hardcoded table names
Works with any LMS (Moodle, Totara, custom) automatically
"""

from typing import Dict, Any, List, Optional
import logging
import json
import ollama
from .mysql_views_adapter import get_view_adapter
from sqlalchemy import create_engine
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class UniversalHayStackPipeline:
    """
    Universal pipeline that works with any LMS using canonical schema
    """
    
    def __init__(self, database_connector):
        self.db = database_connector
        
        # Initialize canonical views adapter
        self._setup_canonical_schema()
        
        # Universal prompts using canonical schema
        self.prompts = self._get_universal_prompts()
    
    def _setup_canonical_schema(self):
        """Setup canonical schema views for current database"""
        try:
            # Get tenant key from database config
            from config import DB_CONFIG
            tenant_key = f"{DB_CONFIG['host']}|{DB_CONFIG['database']}"
            
            # Create engine for views adapter
            connection_string = (
                f"mysql+pymysql://{DB_CONFIG['user']}:{quote_plus(DB_CONFIG['password'])}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
                f"?charset=utf8mb4"
            )
            
            engine = create_engine(connection_string, pool_pre_ping=True)
            
            # Get views adapter (auto-detects schema type)
            self.views_adapter = get_view_adapter(engine, tenant_key, schema_type="auto")
            
            # Ensure canonical views exist
            existing_views = self.views_adapter.list_existing_views()
            if not existing_views:
                logger.info("🏗️ Creating canonical schema views...")
                create_result = self.views_adapter.create_canonical_views()
                if create_result["errors"]:
                    logger.error(f"❌ Failed to create views: {create_result['errors']}")
                else:
                    logger.info(f"✅ Created canonical views: {create_result['created_views']}")
            
            # Validate views
            validation = self.views_adapter.validate_canonical_views()
            if not validation["overall_valid"]:
                logger.error("❌ Canonical views validation failed")
            else:
                logger.info("✅ Canonical schema ready")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup canonical schema: {e}")
            self.views_adapter = None
    
    def _get_universal_prompts(self) -> Dict[str, str]:
        """Get universal prompts using canonical schema"""
        return {
            "intent_classification": """Classify the user's intent based on their query about LMS data.

AVAILABLE INTENTS:
1. general_stats - General statistics, counts, summaries
2. course_info - Specific course details, enrollment info  
3. user_progress - Individual user progress, completions
4. enrollment_analysis - Enrollment patterns, methods, trends
5. role_permissions - User roles, permissions, access levels
6. comparative_analysis - Comparisons between courses, users, time periods

KEYWORDS FOR EACH INTENT:
- general_stats: "how many", "total", "count", "statistics", "overview", "summary"
- course_info: "course details", "course enrollment", "course participants", specific course names
- user_progress: "progress", "completion", "finished", "completed", "user performance"  
- enrollment_analysis: "enrollment trends", "enrollment methods", "when enrolled", "enrollment patterns"
- role_permissions: "role", "permission", "access", "teacher", "student", "instructor", "admin"
- comparative_analysis: "compare", "versus", "difference between", "trend over time"

Return only the intent name, nothing else.""",

            "sql_generation": """You are an expert SQL query generator for Universal LMS databases.
Generate ONLY safe SELECT queries using the canonical schema provided.

🌟 **UNIVERSAL CANONICAL SCHEMA** - Use ONLY these tables:

📋 **canon_user** - Universal user table
   - Columns: id, username, first_name, last_name, email, deleted, created_at
   - Purpose: All users/students/learners in the LMS
   - Key filter: WHERE deleted = 0 for active users

📋 **canon_course** - Universal course table  
   - Columns: id, name, code, visible, summary, created_at
   - Purpose: All courses/classes/modules offered
   - Key filter: WHERE visible = 1 for active courses

📋 **canon_enrollment** - Universal enrollment table
   - Columns: user_id, course_id, status, enrollment_method, created_at, start_date, end_date
   - Purpose: User enrollments in courses
   - Key relationships: JOIN canon_user ON id = user_id, JOIN canon_course ON id = course_id
   - Key filter: WHERE status = 0 for active enrollments

📋 **canon_role_assignment** - Universal roles table
   - Columns: user_id, role, context_type, context_id, created_at
   - Purpose: User roles and permissions  
   - Key relationships: JOIN canon_user ON id = user_id
   - Key values: context_type IN ('course', 'system'), role IN ('student', 'teacher', 'admin')

🚨 **CRITICAL RULES:**
- Use ONLY the canonical table names above (canon_user, canon_course, etc.)
- NEVER reference physical table names (ttl_, mdl_, etc.) 
- All column names are exactly as shown above
- Always include appropriate WHERE clauses for data filtering
- Use proper JOINs between canonical tables

⏰ **TIMESTAMP HANDLING:**
- created_at fields are in standard MySQL DATETIME format
- Use standard date functions: YEAR(created_at), DATE(created_at)
- For date ranges: WHERE created_at >= '2024-01-01'

🔍 **EXAMPLE PATTERNS:**

1. **User Statistics:**
```sql
SELECT 
    COUNT(*) as total_users,
    COUNT(CASE WHEN deleted = 0 THEN 1 END) as active_users
FROM canon_user
```

2. **Course Enrollment Counts:**
```sql
SELECT 
    c.name,
    COUNT(e.user_id) as enrollment_count
FROM canon_course c
LEFT JOIN canon_enrollment e ON c.id = e.course_id AND e.status = 0
WHERE c.visible = 1
GROUP BY c.id, c.name
ORDER BY enrollment_count DESC
```

3. **User Course Progress:**
```sql
SELECT 
    u.first_name,
    u.last_name,
    c.name as course_name,
    e.status
FROM canon_user u
JOIN canon_enrollment e ON u.id = e.user_id
JOIN canon_course c ON c.id = e.course_id  
WHERE u.deleted = 0 AND c.visible = 1
```

4. **Role Analysis:**
```sql
SELECT 
    ra.role,
    COUNT(*) as user_count
FROM canon_role_assignment ra
JOIN canon_user u ON u.id = ra.user_id
WHERE u.deleted = 0 AND ra.context_type = 'course'
GROUP BY ra.role
```

Generate clean, efficient SQL using only the canonical schema above.
""",

            "response_generation": """Generate a helpful response based on the SQL query results.

GUIDELINES:
- Use clear, conversational language
- Include specific numbers and statistics
- Format large numbers with commas (e.g., 1,234 users)
- Use emojis sparingly for visual appeal
- Provide context and interpretation of the data
- Suggest follow-up questions when appropriate

RESPONSE STRUCTURE:
1. Direct answer to the question
2. Key statistics/findings  
3. Brief interpretation or insight
4. Optional: Related information or suggestions

Keep responses concise but informative."""
        }
    
    def process_query(self, query: str, user_context: Dict[str, Any]) -> str:
        """Process a query using canonical schema"""
        try:
            user_id = user_context.get('user_id')
            if not user_id:
                return "Error: User authentication required"
            
            # Check if canonical views are available
            if not self.views_adapter:
                return "Error: Canonical schema not available. Please contact administrator."
            
            logger.info(f"🔍 Processing query with canonical schema: {query}")
            
            # Step 1: Classify intent
            intent = self._classify_intent(query)
            logger.info(f"📋 Classified intent: {intent}")
            
            # Step 2: Generate SQL using canonical schema
            sql_query = self._generate_sql(query, intent, user_context)
            logger.info(f"🔧 Generated SQL: {sql_query}")
            
            if not sql_query:
                return "I couldn't generate an appropriate query for your request."
            
            # Step 3: Execute query
            result = self.db.execute_query_df(sql_query)
            
            if result.empty:
                return "No data found matching your criteria."
            
            # Step 4: Generate response
            response = self._generate_response(query, sql_query, result, intent)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return f"Error processing query: {str(e)}"
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intent using LLM"""
        try:
            response = ollama.chat(
                model="llama3.1:8b",
                messages=[{
                    'role': 'system',
                    'content': self.prompts["intent_classification"]
                }, {
                    'role': 'user', 
                    'content': query
                }]
            )
            
            intent = response['message']['content'].strip().lower()
            
            # Validate intent
            valid_intents = [
                'general_stats', 'course_info', 'user_progress', 
                'enrollment_analysis', 'role_permissions', 'comparative_analysis'
            ]
            
            if intent in valid_intents:
                return intent
            else:
                return 'general_stats'  # Default fallback
                
        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}")
            return 'general_stats'
    
    def _generate_sql(self, query: str, intent: str, user_context: Dict[str, Any]) -> Optional[str]:
        """Generate SQL using canonical schema"""
        try:
            # Add user context to the prompt
            context_info = f"""
USER CONTEXT:
- User ID: {user_context.get('user_id', 'Unknown')}
- Query Intent: {intent}
- Original Query: "{query}"

Generate a SQL query to answer this question using the canonical schema.
Return ONLY the SQL query, no explanations or markdown formatting.
"""
            
            response = ollama.chat(
                model="llama3.1:8b",
                messages=[{
                    'role': 'system',
                    'content': self.prompts["sql_generation"]
                }, {
                    'role': 'user',
                    'content': context_info
                }]
            )
            
            sql = response['message']['content'].strip()
            
            # Clean up the SQL (remove markdown formatting if present)
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # Basic SQL validation
            if not sql.upper().startswith('SELECT'):
                logger.warning(f"⚠️ Generated query doesn't start with SELECT: {sql[:50]}...")
                return None
            
            # Ensure it uses canonical tables
            if not any(table in sql.lower() for table in ['canon_user', 'canon_course', 'canon_enrollment', 'canon_role_assignment']):
                logger.warning(f"⚠️ Query doesn't use canonical tables: {sql[:50]}...")
                return None
            
            return sql
            
        except Exception as e:
            logger.error(f"❌ SQL generation failed: {e}")
            return None
    
    def _generate_response(self, original_query: str, sql_query: str, results, intent: str) -> str:
        """Generate natural language response"""
        try:
            # Convert results to a summary
            results_summary = self._summarize_results(results)
            
            prompt_content = f"""
ORIGINAL QUESTION: "{original_query}"
QUERY INTENT: {intent}  
SQL EXECUTED: {sql_query}
RESULTS SUMMARY: {results_summary}

Generate a helpful, conversational response that answers the user's question.
"""
            
            response = ollama.chat(
                model="llama3.1:8b", 
                messages=[{
                    'role': 'system',
                    'content': self.prompts["response_generation"]
                }, {
                    'role': 'user',
                    'content': prompt_content
                }]
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return f"Query executed successfully. Found {len(results)} results."
    
    def _summarize_results(self, results) -> str:
        """Create a summary of query results"""
        try:
            if results.empty:
                return "No results found."
            
            summary_parts = [
                f"Total rows: {len(results)}",
                f"Columns: {', '.join(results.columns.tolist())}"
            ]
            
            # Add sample data for small result sets
            if len(results) <= 10:
                sample_data = results.head(5).to_dict('records')
                summary_parts.append(f"Sample data: {sample_data}")
            else:
                summary_parts.append(f"First few rows: {results.head(3).to_dict('records')}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"❌ Results summarization failed: {e}")
            return f"Results contain {len(results)} rows"
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the canonical schema"""
        if self.views_adapter:
            return self.views_adapter.get_canonical_schema_info()
        else:
            return {"error": "Canonical schema not available"}


# Backward compatibility - existing code will work
class HayStackPipeline(UniversalHayStackPipeline):
    """Backward compatible alias"""
    pass