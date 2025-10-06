<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

/**
 * Class providing completions for HayStack MCP integration
 * Modified to use HayStack MCP instead of Azure OpenAI
 *
 * @package    block_openai_chat
 * @copyright  2024 Human Logic - HayStack Integration
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
*/

namespace block_openai_chat\completion;

use block_openai_chat\completion;
defined('MOODLE_INTERNAL') || die;

// Include HayStack integration class
require_once(__DIR__ . '/../../haystack_integration.php');

class haystack_assistant extends \block_openai_chat\completion {

    private $thread_id;
    private $haystack_integration;

    public function __construct($model, $message, $history, $block_settings, $thread_id) {
        parent::__construct($model, $message, $history, $block_settings);

        // Initialize HayStack integration - Configure in Moodle block settings
        $haystack_endpoint = get_config('block_openai_chat', 'haystack_endpoint') ?: getenv('HAYSTACK_ENDPOINT') ?: 'http://your-haystack-server:8000';
        $tenant_key = get_config('block_openai_chat', 'haystack_tenant') ?: 'moodle_demos7_remote';
        
        $this->haystack_integration = new \block_openai_chat\completion\haystack_integration($haystack_endpoint, $tenant_key);

        // Generate or use existing thread ID
        if (!$thread_id) {
            $this->thread_id = $this->generate_thread_id();
        } else {
            $this->thread_id = $thread_id;
        }
        
        // Ensure HayStack tenant is configured
        $this->ensure_tenant_configured();
    }

    /**
     * Create completion using HayStack MCP instead of OpenAI
     * @return array The response from HayStack MCP
     */
    public function create_completion($context) {
        global $USER, $DB;
        
        // Get user role information
        $user_role = $this->get_user_role($USER->id);
        
        // Check if HayStack is available
        if (!$this->haystack_integration->check_health()) {
            // Log the issue with server details
            $endpoint = get_config('block_openai_chat', 'haystack_endpoint') ?: 'not configured';
            error_log("HayStack server unavailable at: $endpoint. Falling back to Azure OpenAI");
            return $this->fallback_to_azure_openai();
        }
        
        // Process query through HayStack MCP
        $haystack_response = $this->haystack_integration->process_query(
            $this->message,
            $USER->id,
            $user_role
        );
        
        if ($haystack_response['success']) {
            return [
                "id" => uniqid('haystack_'),
                "message" => $this->haystack_integration->format_response($haystack_response),
                "thread_id" => $this->thread_id,
                "source" => "haystack_mcp",
                "metadata" => $haystack_response['metadata'] ?? []
            ];
        } else {
            // Handle HayStack errors gracefully
            error_log("HayStack processing error: " . ($haystack_response['error'] ?? 'Unknown error'));
            
            return [
                "id" => uniqid('error_'),
                "message" => $haystack_response['message'] ?? "I'm sorry, I couldn't process your request right now.",
                "thread_id" => $this->thread_id,
                "source" => "error",
                "error" => $haystack_response['error'] ?? null
            ];
        }
    }

    /**
     * Determine user role for RBAC
     */
    private function get_user_role($user_id) {
        global $DB;
        
        // Check for manager role first
        $sql = '
            SELECT 1
            FROM {role_assignments} AS ra
            JOIN {role} AS r ON r.id = ra.roleid
            WHERE ra.userid = :userid AND r.shortname = :shortname
        ';
        
        $params = ['userid' => $user_id, 'shortname' => 'manager'];
        if ($DB->record_exists_sql($sql, $params)) {
            return 'manager';
        }
        
        // Check for admin role
        $params['shortname'] = 'admin';
        if ($DB->record_exists_sql($sql, $params)) {
            return 'admin';
        }
        
        // Check for teacher roles
        $teacher_roles = ['editingteacher', 'teacher'];
        foreach ($teacher_roles as $role) {
            $params['shortname'] = $role;
            if ($DB->record_exists_sql($sql, $params)) {
                return 'teacher';
            }
        }
        
        // Check for student role
        $params['shortname'] = 'student';
        if ($DB->record_exists_sql($sql, $params)) {
            return 'student';
        }
        
        // Default to student if no specific role found
        return 'student';
    }
    
    /**
     * Ensure HayStack tenant is properly configured
     */
    private function ensure_tenant_configured() {
        global $CFG;
        
        // Get Moodle database configuration
        $db_config = [
            'host' => $CFG->dbhost,
            'port' => !empty($CFG->dboptions['dbport']) ? $CFG->dboptions['dbport'] : 3306,
            'database' => $CFG->dbname,
            'username' => $CFG->dbuser,
            'password' => $CFG->dbpass
        ];
        
        // Configure tenant in HayStack (idempotent operation)
        $success = $this->haystack_integration->configure_tenant($db_config);
        
        if (!$success) {
            error_log("Warning: Failed to configure HayStack tenant for Moodle database");
        }
    }

    /**
     * Generate thread ID for session tracking
     */
    private function generate_thread_id() {
        global $USER;
        return 'moodle_user_' . $USER->id . '_session_' . time();
    }
    
    /**
     * Fallback to original Azure OpenAI implementation
     * This preserves your existing functionality as backup
     */
    private function fallback_to_azure_openai() {
        global $USER, $DB;
        
        // Get Azure configuration from Moodle settings (configure these in your Moodle block)
        $azure_function_key = get_config('block_openai_chat', 'azure_function_key') ?: 'CONFIGURE_IN_MOODLE_SETTINGS';
        $azure_function_url = get_config('block_openai_chat', 'azure_function_url') ?: 'https://your-azure-function.azurewebsites.net/api/getadmininfo';
        
        // Get user role for the original implementation
        $sql = '
            SELECT 1
            FROM {role_assignments} AS ra
            JOIN {role} AS r ON r.id = ra.roleid
            WHERE ra.userid = :userid AND r.shortname = :shortname
        ';
       
        $params = [
            'userid' => $USER->id,
            'shortname' => 'manager'
        ];

        $hasAllowedRole = $DB->record_exists_sql($sql, $params);
        if($hasAllowedRole){
            $user_role = 'manager';
        } else {
            $user_role = 'admin';
        }
        
        $curlbody = [
            "thread_id" => $this->thread_id,
            "useremail" => $USER->email,
            "message" => $this->message,
            "role" => $user_role
        ];

        $curl = curl_init();
        curl_setopt($curl, CURLOPT_SSL_VERIFYPEER, false);
        curl_setopt_array($curl, array(
            CURLOPT_URL => $azure_function_url . '?code=' . $azure_function_key,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_ENCODING => '',
            CURLOPT_MAXREDIRS => 10,
            CURLOPT_TIMEOUT => 0,
            CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
            CURLOPT_CUSTOMREQUEST => 'POST',
            CURLOPT_POSTFIELDS => json_encode($curlbody),
            CURLOPT_HTTPHEADER => array(
                'Content-Type: application/json'
            ),
        ));

        $response = curl_exec($curl);
        curl_close($curl);
        
        if (!$response) {
            return [
                "id" => uniqid('fallback_error_'),
                "message" => "I'm sorry, I'm experiencing technical difficulties. Please try again later.",
                "thread_id" => $this->thread_id
            ];
        }
        
        $response = json_decode($response);
        
        // Process message_text to convert markdown links to HTML anchor tags
        $message_text = $response->message_text ?? "Sorry, I couldn't process your request.";
        
        // Check if there's a file_download_url and markdown-style links in the message
        if (isset($response->file_download_url) && !empty($response->file_download_url)) {
            $pattern = '/\[([^\]]+)\]\(sandbox:[^)]+\)/';
            $message_text = preg_replace($pattern, '<a href="' . $response->file_download_url . '" target="_blank">$1</a>', $message_text);
        }
        
        return [
            "id" => $response->rud_id ?? uniqid('fallback_'),
            "message" => $message_text,
            "thread_id" => $response->thread_id ?? $this->thread_id,
            "source" => "azure_openai_fallback"
        ];
    }
}

// Also create a compatibility class that extends the original assistant
// This maintains backward compatibility while using HayStack by default
class assistant extends haystack_assistant {
    // This maintains backward compatibility while using HayStack by default
}