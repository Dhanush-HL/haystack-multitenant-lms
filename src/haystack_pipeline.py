""""""

Haystack Pipeline implementation for LMS integrationHaystack Pipeline with LLM Brain Integration

"""Orchestrates the entire flow from natural language query to SQL generation and response creation

import logging"""

from typing import Dict, Any, List, Optional

import jsonfrom typing import Dict, List, Any, Optional

import logging

logger = logging.getLogger(__name__)from dataclasses import dataclass

from haystack import Pipeline, component

class HayStackPipeline:from haystack.components.generators import OpenAIGenerator

    """Main pipeline for processing LMS queries"""from haystack.components.builders import PromptBuilder

    import json

    def __init__(self, database_connector, vector_store):import ollama

        self.db = database_connector

        self.vector_store = vector_storefrom database_connector import TotaraLMSConnector, QueryResult, DatabaseConfig

        

    def process_query(self, query: str, user_context: Dict[str, Any]) -> str:logger = logging.getLogger(__name__)

        """Process a user query and return response"""

        try:

            user_id = user_context.get('user_id')@dataclass

            if not user_id:class QueryContext:

                return "Error: User authentication required"    """Context for query processing"""

                user_id: str

            # Determine query type and route accordingly    session_id: str

            query_lower = query.lower()    user_query: str

                conversation_history: List[Dict[str, str]]

            if 'course' in query_lower and ('enrolled' in query_lower or 'my' in query_lower):    user_profile: Optional[Dict[str, Any]] = None

                return self._handle_courses_query(user_id)

            elif 'assignment' in query_lower:

                return self._handle_assignments_query(user_id)@dataclass

            elif 'grade' in query_lower:class ProcessedResponse:

                return self._handle_grades_query(user_id)    """Processed response from the pipeline"""

            else:    response_text: str

                return self._handle_general_query(query, user_context)    sql_query: Optional[str]

                    data: Optional[List[Dict[str, Any]]]

        except Exception as e:    confidence: float

            logger.error(f"❌ Pipeline error: {e}")    session_id: str

            return f"Error processing query: {str(e)}"    requires_followup: bool = False

    

    def _handle_courses_query(self, user_id: int) -> str:

        """Handle courses-related queries"""@component

        try:class CustomOllamaGenerator:

            courses = self.db.get_user_courses(user_id)    """Custom Ollama generator with streaming support"""

                

            if not courses:    def __init__(self, model: str = "llama3.1:8b", url: str = "http://localhost:11434", **kwargs):

                return "You are not currently enrolled in any courses."        self.model = model

                    self.url = url

            response = f"📚 **Enrolled Courses ({len(courses)}):**\n"        self.generation_kwargs = kwargs.get('generation_kwargs', {})

            for i, course in enumerate(courses, 1):        self.stream_callback = kwargs.get('stream_callback', None)

                response += f"{i}. **{course['fullname']}**"    

                if course['shortname']:    @component.output_types(replies=List[str])

                    response += f" Code: {course['shortname']}"    def run(self, prompt: str) -> Dict[str, Any]:

                response += "\n"        """Generate response using Ollama with optional streaming"""

                    try:

            return response            # Check if streaming is requested

                        if self.stream_callback:

        except Exception as e:                return self._run_streaming(prompt)

            logger.error(f"❌ Courses query error: {e}")            else:

            return "Error retrieving courses information."                return self._run_non_streaming(prompt)

            except Exception as e:

    def _handle_assignments_query(self, user_id: int) -> str:            logger.error(f"Error generating response with Ollama: {e}")

        """Handle assignments-related queries"""            return {

        try:                "replies": [f"Error: {str(e)}"]

            assignments = self.db.get_user_assignments(user_id)            }

                

            if not assignments:    def _run_streaming(self, prompt: str) -> Dict[str, Any]:

                return "You have no assignments at this time."        """Generate streaming response"""

                    try:

            response = f"📝 **Your Assignments ({len(assignments)}):**\n"            full_response = ""

            for i, assignment in enumerate(assignments, 1):            stream = ollama.generate(

                response += f"{i}. **{assignment['name']}** ({assignment['course_name']})"                model=self.model,

                if assignment['duedate']:                prompt=prompt,

                    response += f" - Due: {assignment['duedate']}"                stream=True,

                if assignment['status']:                options={

                    response += f" - Status: {assignment['status']}"                    'temperature': self.generation_kwargs.get('temperature', 0.3),

                response += "\n"                    'top_p': self.generation_kwargs.get('top_p', 0.9),

                                'num_predict': self.generation_kwargs.get('max_tokens', 1000)

            return response                }

                        )

        except Exception as e:            

            logger.error(f"❌ Assignments query error: {e}")            # Stream tokens through callback

            return "Error retrieving assignments information."            for chunk in stream:

                    if 'response' in chunk:

    def _handle_grades_query(self, user_id: int) -> str:                    token = chunk['response']

        """Handle grades-related queries"""                    full_response += token

        try:                    if self.stream_callback:

            grades = self.db.get_user_grades(user_id)                        self.stream_callback(token)

                        

            if not grades:            return {

                return "No grades are currently available."                "replies": [full_response]

                        }

            response = f"📊 **Your Grades ({len(grades)}):**\n"        except Exception as e:

            for i, grade in enumerate(grades, 1):            logger.error(f"Error in streaming generation: {e}")

                response += f"{i}. **{grade['course_name']}** - {grade['itemname']}: "            return {

                if grade['final_grade'] and grade['max_grade']:                "replies": [f"Streaming error: {str(e)}"]

                    response += f"{grade['final_grade']}/{grade['max_grade']}"            }

                response += "\n"    

                def _run_non_streaming(self, prompt: str) -> Dict[str, Any]:

            return response        """Generate non-streaming response"""

                    try:

        except Exception as e:            # Use the ollama package directly

            logger.error(f"❌ Grades query error: {e}")            response = ollama.generate(

            return "Error retrieving grades information."                model=self.model,

                    prompt=prompt,

    def _handle_general_query(self, query: str, user_context: Dict[str, Any]) -> str:                options={

        """Handle general queries"""                    'temperature': self.generation_kwargs.get('temperature', 0.3),

        # For now, provide a helpful response                    'top_p': self.generation_kwargs.get('top_p', 0.9),

        return (                    'num_predict': self.generation_kwargs.get('max_tokens', 1000)

            "I can help you with:\n"                }

            "• 📚 Your enrolled courses - ask 'What are my courses?'\n"            )

            "• 📝 Your assignments - ask 'Show me my assignments'\n"            

            "• 📊 Your grades - ask 'What are my grades?'\n\n"            return {

            "Please try asking about one of these topics!"                "replies": [response['response']]

        )            }
        except Exception as e:
            logger.error(f"Error in non-streaming generation: {e}")
            return {
                "replies": [f"Generation error: {str(e)}"]
            }


class LLMBrain:
    """
    Central LLM Brain that controls the entire conversation flow
    Acts as the orchestrator for all operations
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.llm = CustomOllamaGenerator(
            model=model_name,
            url="http://localhost:11434",
            generation_kwargs={
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        )
        
        # System prompts for different tasks
        self.system_prompts = {
            "intent_analysis": """You are an AI assistant specialized in understanding user intents for a Totara LMS (Learning Management System) database. 
Analyze the user's query and determine:
1. What information they're seeking
2. What database tables might be involved
3. Whether this requires SQL query generation
4. The complexity level (simple lookup, complex analysis, etc.)

You must respond with ONLY a JSON object in this exact format, no other text:
{
    "intent": "description of what user wants",
    "requires_sql": true/false,
    "tables_involved": ["table1", "table2"],
    "complexity": "simple|medium|complex",
    "query_type": "user_info|course_info|enrollment|completion|user_role|general_stats|other"
}

QUERY TYPE DEFINITIONS:
- user_info: Basic user details (name, email, etc.)
- course_info: Course details and information
- enrollment: How/when users enrolled in courses (enrollment methods, dates)
- completion: Course completion status and progress
- user_role: What permissions/roles users have in courses (student, teacher, admin)
- general_stats: Statistics and aggregated data
- other: General inquiries not requiring database queries

KEYWORDS FOR ROLE QUERIES:
- "role", "permission", "access", "teacher", "student", "instructor", "admin"
- "what is the role", "what role does", "permissions in", "access level"
""",

            "sql_generation": """You are an expert SQL query generator for Totara LMS database. 
Generate ONLY safe SELECT queries based on the user's intent and schema provided.

CRITICAL: Use ONLY these EXACT table names (no variations allowed):
- ttl_user (NOT users, ttl_users, or user)
- ttl_course (NOT courses, ttl_courses, or course)  
- ttl_user_enrolments (NOT course_enrollments, user_enrollments, or enrollments)
- ttl_enrol (NOT enrol, enrollment, or enroll)
- ttl_course_completions (NOT course_completions, completions, or completion)
- ttl_role_assignments (for user roles in courses)
- ttl_role (for role definitions)
- ttl_context (for course contexts)

REQUIRED TABLE STRUCTURE:
- ttl_user: id, username, firstname, lastname, email, deleted (WHERE deleted = 0 for active users)
- ttl_course: id, fullname, shortname, summary, visible
- ttl_user_enrolments: id, enrolid, userid, status, timecreated, timestart, timeend
- ttl_enrol: id, courseid, enrol (enrollment method: manual, self, cohort, etc.)
- ttl_course_completions: id, userid, course, timecompleted, timestarted, timeenrolled
- ttl_role_assignments: id, roleid, contextid, userid (links users to roles)
- ttl_role: id, name, shortname (role definitions: student, editingteacher, teacher, etc.)
- ttl_context: id, contextlevel, instanceid (contextlevel=50 for courses, instanceid=courseid)

TIMESTAMP HANDLING:
- ALL timestamps in Totara are UNIX timestamps (integer seconds since 1970)
- MUST use FROM_UNIXTIME(timestamp_field) for date operations
- For year extraction: YEAR(FROM_UNIXTIME(timestamp_field))
- For date comparison: FROM_UNIXTIME(timestamp_field) >= '2024-01-01'

IMPORTANT DISTINCTION:
- ENROLLMENT METHOD (ttl_enrol.enrol): HOW user was enrolled (manual, self, etc.)
- USER ROLE (ttl_role_assignments): WHAT permissions user has (student, teacher, etc.)

EXAMPLE QUERIES FOR COMMON PATTERNS:

1. COURSE ENROLLMENT COUNTS (general_stats):
SELECT c.fullname, COUNT(*) as enrollment_count
FROM ttl_user_enrolments ue
JOIN ttl_enrol e ON ue.enrolid = e.id
JOIN ttl_course c ON e.courseid = c.id
JOIN ttl_user u ON ue.userid = u.id
WHERE u.deleted = 0 
AND YEAR(FROM_UNIXTIME(ue.timecreated)) = 2024
GROUP BY c.id, c.fullname
ORDER BY enrollment_count DESC
LIMIT 1

2. USER ENROLLMENTS (enrollment):
SELECT c.fullname, c.shortname, ue.status, e.enrol as enrollment_method
FROM ttl_user_enrolments ue
JOIN ttl_enrol e ON ue.enrolid = e.id
JOIN ttl_course c ON e.courseid = c.id
JOIN ttl_user u ON ue.userid = u.id
WHERE u.id = 71 AND u.deleted = 0

3. USER ROLES (user_role):
SELECT c.fullname, r.name as role_name, r.shortname as role_code
FROM ttl_role_assignments ra
JOIN ttl_context ctx ON ra.contextid = ctx.id
JOIN ttl_course c ON ctx.instanceid = c.id
JOIN ttl_role r ON ra.roleid = r.id
JOIN ttl_user u ON ra.userid = u.id
WHERE u.id = 71 AND u.deleted = 0 AND ctx.contextlevel = 50

4. USER INFO (user_info):
SELECT u.firstname, u.lastname, u.email 
FROM ttl_user u 
WHERE u.id = 71 AND u.deleted = 0

RULES:
- Only SELECT statements allowed
- Always JOIN ttl_user table and include WHERE u.deleted = 0
- Use proper JOINs between related tables
- Use specific values (like user IDs or usernames) directly in WHERE clauses
- Use LIMIT for performance with large result sets
- For ROLE queries: Use ttl_role_assignments, ttl_role, ttl_context tables
- For ENROLLMENT queries: Use ttl_user_enrolments, ttl_enrol tables
- For role queries, include ctx.contextlevel = 50 (course context)
- ALWAYS use FROM_UNIXTIME() for timestamp operations

You must respond with ONLY the SQL query, no explanations, no code blocks, no other text.

Database Schema: {schema_info}
User Intent: {intent}
User Query: {user_query}

SQL Query:""",

            "response_generation": """You are a friendly, knowledgeable university chatbot assistant. 
Create a conversational response based on the query results and context.

Context:
- User Query: {user_query}
- SQL Query Used: {sql_query}
- Data Retrieved: {query_data}
- User Profile: {user_profile}

Create a natural, helpful response that:
1. Directly answers the user's question
2. Explains the data in an easy-to-understand way
3. Offers relevant follow-up suggestions
4. Maintains a friendly, professional tone

Response:"""
        }

    async def analyze_intent(self, query_context: QueryContext) -> Dict[str, Any]:
        """Analyze user intent and determine processing strategy"""
        prompt = f"""{self.system_prompts['intent_analysis']}

IMPORTANT: Respond with ONLY the JSON object, no other text, no code blocks, no explanations.

User Query: {query_context.user_query}"""
        
        try:
            result = self.llm.run(prompt=prompt)
            response_text = result["replies"][0] if result["replies"] else "{}"
            
            # Clean the response - remove code blocks and extra text if present
            response_text = response_text.strip()
            
            # Extract JSON from code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            # If response starts with explanation text, try to find JSON
            if response_text.startswith("Here") or response_text.startswith("The"):
                # Look for JSON object starting with {
                start_pos = response_text.find("{")
                if start_pos != -1:
                    # Find the matching closing brace
                    brace_count = 0
                    end_pos = start_pos
                    for i, char in enumerate(response_text[start_pos:], start_pos):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    response_text = response_text[start_pos:end_pos]
            
            # Parse JSON response
            intent_data = json.loads(response_text)
            return intent_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing intent analysis: {e}")
            logger.error(f"Raw response: {response_text}")
            return {
                "intent": "general inquiry",
                "requires_sql": False,
                "tables_involved": [],
                "complexity": "simple",
                "query_type": "other"
            }

    async def generate_sql_query(self, query_context: QueryContext, 
                                intent_data: Dict[str, Any], 
                                schema_info: str) -> Optional[str]:
        """Generate SQL query based on intent and schema"""
        if not intent_data.get("requires_sql", False):
            return None
            
        prompt = self.system_prompts["sql_generation"].format(
            schema_info=schema_info,
            intent=intent_data["intent"],
            user_query=query_context.user_query
        )
        
        try:
            result = self.llm.run(prompt=prompt)
            sql_query = result["replies"][0] if result["replies"] else ""
            
            # Clean up the response to extract just the SQL
            sql_query = sql_query.strip()
            
            # Remove code blocks if present
            if "```sql" in sql_query:
                start = sql_query.find("```sql") + 6
                end = sql_query.find("```", start)
                if end != -1:
                    sql_query = sql_query[start:end].strip()
            elif "```" in sql_query:
                start = sql_query.find("```") + 3
                end = sql_query.find("```", start)
                if end != -1:
                    sql_query = sql_query[start:end].strip()
            
            # If response contains explanatory text, try to extract SQL
            if sql_query and not sql_query.upper().startswith('SELECT'):
                # Look for SELECT statement
                select_pos = sql_query.upper().find('SELECT')
                if select_pos != -1:
                    sql_query = sql_query[select_pos:]
                    # Find end of SQL statement (look for semicolon or double newline)
                    end_pos = sql_query.find(';')
                    if end_pos == -1:
                        # Look for where explanatory text might start
                        patterns = ['\n\nIn this query:', '\n\nNote:', '\n\nExplanation:', '\n\n*']
                        for pattern in patterns:
                            pos = sql_query.find(pattern)
                            if pos != -1:
                                sql_query = sql_query[:pos]
                                break
                    else:
                        sql_query = sql_query[:end_pos]
            
            # Clean up the final query
            sql_query = sql_query.strip()
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1]
            
            return sql_query if sql_query else None
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return None

    async def generate_response(self, query_context: QueryContext,
                               sql_query: Optional[str],
                               query_result: Optional[QueryResult]) -> str:
        """Generate final conversational response"""
        query_data = "No data retrieved" if not query_result or not query_result.success else str(query_result.data)
        
        prompt = self.system_prompts["response_generation"].format(
            user_query=query_context.user_query,
            sql_query=sql_query or "No SQL query used",
            query_data=query_data,
            user_profile=query_context.user_profile or "No profile available"
        )
        
        try:
            result = self.llm.run(prompt=prompt)
            return result["replies"][0] if result["replies"] else "I apologize, but I couldn't generate a proper response."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request."


class HaystackPipeline:
    """
    Main Haystack pipeline that orchestrates the entire flow
    """
    
    def __init__(self, db_config: DatabaseConfig, model_name: str = "llama3.1:8b"):
        self.db_connector = TotaraLMSConnector(db_config)
        self.llm_brain = LLMBrain(model_name)
        self.pipeline = None
        self._build_pipeline()

    def _build_pipeline(self):
        """Build the Haystack pipeline components"""
        # Create pipeline components
        intent_analyzer = PromptBuilder(
            template=self.llm_brain.system_prompts["intent_analysis"] + "\n\nUser Query: {query}"
        )
        
        sql_generator = PromptBuilder(
            template=self.llm_brain.system_prompts["sql_generation"]
        )
        
        response_generator = PromptBuilder(
            template=self.llm_brain.system_prompts["response_generation"]
        )
        
        # Build the pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("intent_analyzer", intent_analyzer)
        self.pipeline.add_component("sql_generator", sql_generator)
        self.pipeline.add_component("response_generator", response_generator)
        self.pipeline.add_component("llm", self.llm_brain.llm)

    async def initialize(self):
        """Initialize pipeline and database connection"""
        await self.db_connector.connect()
        logger.info("Haystack pipeline initialized successfully")

    async def process_query(self, query_context: QueryContext) -> ProcessedResponse:
        """
        Main processing method that handles the entire flow
        """
        try:
            # Step 1: Analyze user intent
            logger.info(f"Analyzing intent for query: {query_context.user_query}")
            intent_data = await self.llm_brain.analyze_intent(query_context)
            
            # Step 2: Generate SQL query if needed
            sql_query = None
            query_result = None
            
            if intent_data.get("requires_sql", False):
                logger.info("Generating SQL query")
                schema_info = self.db_connector.get_schema_context()
                sql_query = await self.llm_brain.generate_sql_query(
                    query_context, intent_data, schema_info
                )
                
                # Step 3: Execute SQL query
                if sql_query and self.db_connector.validate_sql_query(sql_query):
                    logger.info(f"Executing SQL query: {sql_query}")
                    query_result = await self.db_connector.execute_query(sql_query)
                else:
                    logger.warning("Invalid or dangerous SQL query blocked")
                    query_result = QueryResult(
                        success=False,
                        data=[],
                        columns=[],
                        row_count=0,
                        error="Invalid or unsafe query"
                    )
            
            # Step 4: Generate conversational response
            logger.info("Generating final response")
            response_text = await self.llm_brain.generate_response(
                query_context, sql_query, query_result
            )
            
            # Calculate confidence based on success of operations
            confidence = 0.9 if (not intent_data.get("requires_sql") or 
                               (query_result and query_result.success)) else 0.6
            
            return ProcessedResponse(
                response_text=response_text,
                sql_query=sql_query,
                data=query_result.data if query_result else None,
                confidence=confidence,
                session_id=query_context.session_id,
                requires_followup=intent_data.get("complexity") == "complex"
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return ProcessedResponse(
                response_text="I apologize, but I encountered an error while processing your request. Please try again.",
                sql_query=None,
                data=None,
                confidence=0.1,
                session_id=query_context.session_id,
                requires_followup=False
            )

    async def cleanup(self):
        """Cleanup resources"""
        await self.db_connector.disconnect()
        logger.info("Pipeline cleanup completed")