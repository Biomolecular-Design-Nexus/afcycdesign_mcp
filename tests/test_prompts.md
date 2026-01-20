# MCP Cyclic Peptide Tools - Test Prompts

## Tool Discovery Tests

### Prompt 1: List All Tools
"What MCP tools are available for cyclic peptides? Give me a brief description of each."

**Expected Response:**
- Should list all 12 tools from cycpep-tools
- Should categorize into sync tools, submit tools, and job management
- Should include brief descriptions of each tool's purpose

### Prompt 2: Tool Details
"Explain how to use the predict_cyclic_structure tool, including all parameters."

**Expected Response:**
- Should explain the tool takes length, optional output_file, rm_aa, add_rg, soft_iters, quiet parameters
- Should mention it's for fast structure prediction (<10 min)
- Should explain parameter meanings and defaults

## Sync Tool Tests

### Prompt 3: Structure Prediction
"Predict the 3D structure of a cyclic peptide with 8 residues"

**Expected Response:**
- Should call predict_cyclic_structure with length=8
- Should return success status with generated sequence and structure file path
- Should complete within reasonable time

### Prompt 4: Sequence Design
"Design a sequence for the cyclic peptide backbone at examples/data/test_backbone.pdb"

**Expected Response:**
- Should call design_cyclic_sequence with the input file
- Should handle file not found gracefully if file doesn't exist
- Should return designed sequence and metrics if file exists

### Prompt 5: File Validation
"Validate if the file examples/data/test_structure.pdb is a valid cyclic peptide"

**Expected Response:**
- Should call validate_cyclic_peptide_file
- Should return validation results including file size, chains, atom count
- Should handle file not found gracefully

### Prompt 6: Error Handling
"Predict structure for a cyclic peptide with 0 residues"

**Expected Response:**
- Should handle invalid length parameter gracefully
- Should return error status with helpful error message
- Should not crash the server

## Submit API Tests

### Prompt 7: Submit Structure Prediction
"Submit a large structure prediction job for a 20-residue cyclic peptide with 200 iterations"

**Expected Response:**
- Should call submit_large_structure_prediction
- Should return job_id and submission confirmation
- Should provide instructions on checking status

### Prompt 8: Submit Binder Design
"Design a cyclic peptide binder for the target protein at examples/data/target.pdb with 10 residues"

**Expected Response:**
- Should call submit_cyclic_binder_design
- Should return job_id for tracking
- Should handle file not found gracefully

### Prompt 9: Check Job Status
"What's the status of job abc12345?"

**Expected Response:**
- Should call get_job_status with job_id
- Should return job not found if invalid ID
- Should return proper status, timestamps for valid jobs

### Prompt 10: Get Job Results
"Show me the results of completed job def67890"

**Expected Response:**
- Should call get_job_result with job_id
- Should return error if job not completed
- Should return full results for completed jobs

### Prompt 11: View Job Logs
"Show the last 30 lines of logs for job ghi01234"

**Expected Response:**
- Should call get_job_log with tail=30
- Should return log lines and total count
- Should handle job not found gracefully

### Prompt 12: List All Jobs
"List all jobs with status 'completed'"

**Expected Response:**
- Should call list_jobs with status='completed'
- Should return filtered list of completed jobs
- Should show job_id, job_name, status, submitted_at

### Prompt 13: Cancel Running Job
"Cancel the running job jkl45678"

**Expected Response:**
- Should call cancel_job with job_id
- Should return success if job is running
- Should return error if job not running or not found

## Batch Processing Tests

### Prompt 14: Batch Structure Prediction
"Generate structures for cyclic peptides of lengths 6, 8, 10, and 12 residues in batch"

**Expected Response:**
- Should call submit_batch_structure_prediction with lengths=[6,8,10,12]
- Should return batch job_id
- Should set reasonable parameters for batch processing

### Prompt 15: Batch Status Check
"Check the status of batch job batch_xyz123"

**Expected Response:**
- Should call get_job_status for the batch job
- Should return batch job status and progress information

## Real-World Scenarios

### Prompt 16: Full Drug Design Workflow
"I want to design a cyclic peptide drug candidate:
1. First predict a 12-residue structure with compact folding
2. Check the validation of the generated structure
3. Submit a high-accuracy binder design against a target protein
Coordinate all these steps for me."

**Expected Response:**
- Should call predict_cyclic_structure with length=12, add_rg=True
- Should call validate_cyclic_peptide_file on the output structure
- Should call submit_cyclic_binder_design using the generated structure
- Should provide job tracking information

### Prompt 17: Structure Optimization Pipeline
"Optimize a cyclic peptide structure:
1. Start with 8 residues, excluding cysteine and methionine
2. Generate 3 different structures in batch
3. Validate all structures
Give me a comprehensive analysis."

**Expected Response:**
- Should call submit_batch_structure_prediction with rm_aa="C,M"
- Should call validate_cyclic_peptide_file for each generated structure
- Should provide comparative analysis of results

### Prompt 18: Virtual Library Screening
"Create a virtual library of cyclic peptides:
1. Generate structures for 5, 7, 9, and 11 residues
2. For each length, create structures both with and without RG constraints
3. Monitor all jobs and summarize when complete"

**Expected Response:**
- Should submit multiple batch jobs with different parameters
- Should provide job tracking for all submissions
- Should explain how to monitor progress and collect results

### Prompt 19: Membrane Permeability Study
"I need to study membrane permeability of cyclic peptides:
1. Generate 3 different 8-residue peptides
2. Validate their structures
3. Submit design refinement for the best structure
Track all jobs and provide status updates."

**Expected Response:**
- Should coordinate multiple structure predictions
- Should validate outputs systematically
- Should submit follow-up design jobs based on results
- Should provide comprehensive job tracking

### Prompt 20: Error Recovery Workflow
"I submitted a job that failed (job failed_123):
1. Check what went wrong by viewing the logs
2. Try to resubmit with corrected parameters
3. Monitor the new job until completion"

**Expected Response:**
- Should call get_job_log to diagnose the failure
- Should suggest parameter corrections based on error logs
- Should submit corrected job and provide tracking

## Server Management Tests

### Prompt 21: Server Status Check
"What's the current status of the cyclic peptide MCP server and what tools are available?"

**Expected Response:**
- Should call get_server_info
- Should return server name, version, available tools
- Should show directory paths and API information

### Prompt 22: Job Queue Management
"Show me all pending and running jobs in the system"

**Expected Response:**
- Should call list_jobs without status filter or with multiple status filters
- Should return comprehensive job listing
- Should categorize by status

## Error Handling Tests

### Prompt 23: Invalid Parameters
"Predict a cyclic peptide structure with -5 residues"

**Expected Response:**
- Should handle negative length gracefully
- Should return clear error message
- Should not crash the server

### Prompt 24: File Not Found
"Design sequence for the backbone at /nonexistent/file.pdb"

**Expected Response:**
- Should handle file not found gracefully
- Should return file not found error message
- Should suggest checking the file path

### Prompt 25: Invalid Job ID
"Get results for job invalid_job_id_123"

**Expected Response:**
- Should handle invalid job ID gracefully
- Should return job not found error
- Should suggest using list_jobs to find valid IDs

## Performance Tests

### Prompt 26: Quick Response Check
"Get server info and validate it responds within 5 seconds"

**Expected Response:**
- Should call get_server_info
- Should respond quickly (sync operation)
- Should return full server information

### Prompt 27: Concurrent Job Submission
"Submit 3 different structure prediction jobs simultaneously for lengths 8, 10, and 12"

**Expected Response:**
- Should handle multiple job submissions
- Should return unique job IDs for each
- Should manage concurrent execution properly

## Integration Tests

### Prompt 28: End-to-End Validation
"Run a complete test of the cyclic peptide pipeline:
1. Generate a structure
2. Validate the output
3. Submit for sequence design
4. Monitor until completion
5. Validate final result"

**Expected Response:**
- Should execute full pipeline systematically
- Should handle each step's outputs as inputs to next
- Should provide comprehensive progress tracking
- Should validate results at each stage

### Prompt 29: Resource Management
"Submit a large computational job and then check system resource usage"

**Expected Response:**
- Should submit appropriate large job
- Should provide job monitoring information
- Should handle resource considerations appropriately

### Prompt 30: Cleanup and Maintenance
"Show me how to clean up old completed jobs and manage the job queue"

**Expected Response:**
- Should demonstrate job management capabilities
- Should provide guidance on job cleanup
- Should show best practices for queue management