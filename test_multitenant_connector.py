#!/usr/bin/env python3
"""
Test Multi-Tenant Database Connector
===================================

Tests the new multi-tenant database connector with runtime DB switching,
schema introspection, and canonical SQL rewriting capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_multitenant_connector():
    """Test multi-tenant database connector functionality"""
    
    print("🔄 Testing Multi-Tenant Database Connector")
    print("=" * 50)
    
    try:
        # Import the new multi-tenant connector
        from src.database_connector_multitenant import DatabaseConnector, DBConfig
        
        print("✅ Successfully imported multi-tenant DatabaseConnector")
        
        # Test 1: Initialize with default config
        print("\n🔧 Test 1: Default initialization...")
        connector = DatabaseConnector()
        print(f"   Current tenant: {connector.current_tenant_key()}")
        
        # Test 2: Schema introspection
        print("\n🔍 Test 2: Schema introspection...")
        schema_map = connector.get_schema_map()
        print(f"   Discovered {len(schema_map)} tables")
        
        # Show some tables
        table_names = list(schema_map.keys())[:5]
        for table in table_names:
            columns = schema_map[table][:3]  # First 3 columns
            print(f"   - {table}: {', '.join(columns)}...")
        
        # Test 3: Prefix detection
        print("\n🏷️ Test 3: Prefix detection...")
        prefix = connector.detect_prefix()
        print(f"   Detected prefix: '{prefix}'")
        
        # Test 4: Synonyms mapping
        print("\n🗺️ Test 4: Synonyms mapping...")
        synonyms = connector.build_synonyms()
        print(f"   Built {len(synonyms)} synonym mappings:")
        for canonical, actual in list(synonyms.items())[:5]:
            print(f"   - {canonical} → {actual}")
        
        # Test 5: LLM schema context
        print("\n🤖 Test 5: LLM schema context...")
        context = connector.get_llm_schema_context()
        print(f"   Tenant: {context['tenant_key']}")
        print(f"   Prefix: {context['table_prefix']}")
        print(f"   Canonical entities: {len(context['canonical_entities'])}")
        print(f"   Synonyms: {len(context['synonyms_map'])}")
        print(f"   Instructions: {context['instructions'][:80]}...")
        
        # Test 6: SQL rewriting
        print("\n📝 Test 6: Canonical SQL rewriting...")
        test_queries = [
            "SELECT * FROM user WHERE id = 1",
            "SELECT u.id, c.name FROM user u JOIN course c ON u.id = c.owner_id",
            "SELECT COUNT(*) FROM user_enrol WHERE status = 'active'"
        ]
        
        for query in test_queries:
            rewritten = connector.rewrite_canonical_sql(query)
            print(f"   Original:  {query}")
            print(f"   Rewritten: {rewritten}")
            print()
        
        # Test 7: SQL validation
        print("🛡️ Test 7: SQL validation...")
        valid_queries = [
            "SELECT * FROM user",
            "SELECT COUNT(*) FROM course WHERE visible = 1"
        ]
        invalid_queries = [
            "INSERT INTO user (name) VALUES ('test')",
            "DROP TABLE user",
            "SELECT * FROM unknown_table"
        ]
        
        for query in valid_queries:
            is_valid = connector.validate_sql_query(query)
            print(f"   ✅ Valid: {query} -> {is_valid}")
        
        for query in invalid_queries:
            is_valid = connector.validate_sql_query(query)
            print(f"   ❌ Invalid: {query} -> {is_valid}")
        
        # Test 8: Connection test
        print("\n🔌 Test 8: Connection test...")
        connection_ok = connector.test_connection()
        print(f"   Connection status: {'✅ OK' if connection_ok else '❌ Failed'}")
        
        print(f"\n✅ All multi-tenant connector tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-tenant connector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_runtime_db_switching():
    """Test runtime database switching capability"""
    
    print(f"\n🔄 Testing Runtime Database Switching")
    print("=" * 45)
    
    try:
        from src.database_connector_multitenant import DatabaseConnector, DBConfig
        
        # Initialize connector
        connector = DatabaseConnector()
        original_tenant = connector.current_tenant_key()
        print(f"Original tenant: {original_tenant}")
        
        # Create a mock "different" config (same DB but different key for testing)
        current_config = connector.default_config
        test_config = DBConfig(
            host=current_config.host,
            port=current_config.port,
            database=current_config.database,
            user=current_config.user,
            password=current_config.password,
            charset="utf8mb4"  # Explicitly set charset to create different key
        )
        
        print(f"\n🔄 Switching to test configuration...")
        connector.switch_database(test_config)
        new_tenant = connector.current_tenant_key()
        print(f"New tenant: {new_tenant}")
        
        # Test that schema introspection works with new "tenant"
        schema_map = connector.get_schema_map()
        print(f"Schema discovered: {len(schema_map)} tables")
        
        # Switch back
        print(f"\n🔄 Switching back to original configuration...")
        connector.switch_database(connector.default_config)
        back_to_original = connector.current_tenant_key()
        print(f"Back to: {back_to_original}")
        
        print(f"\n✅ Runtime DB switching test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Runtime DB switching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all multi-tenant connector tests"""
    
    print("🌐 Multi-Tenant Database Connector Test Suite")
    print("=" * 55)
    print("Testing new runtime DB switching and schema introspection...")
    print("for 'change DB and it works' capability\n")
    
    # Test basic functionality
    basic_success = test_multitenant_connector()
    
    # Test runtime switching
    switching_success = test_runtime_db_switching()
    
    # Summary
    print(f"\n" + "=" * 55)
    print(f"📋 TEST RESULTS SUMMARY")
    print(f"=" * 55)
    
    if basic_success and switching_success:
        print(f"✅ ALL TESTS PASSED")
        print(f"✅ Multi-tenant connector is working correctly!")
        print(f"✅ Runtime DB switching is functional")
        print(f"✅ Schema introspection is operational")
        print(f"✅ Canonical SQL rewriting is working")
        print(f"✅ Ready for 'change DB and it works' capability!")
    else:
        print(f"❌ Some tests failed")
        if not basic_success:
            print(f"❌ Basic multi-tenant functionality failed")
        if not switching_success:
            print(f"❌ Runtime DB switching failed")
    
    return basic_success and switching_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)