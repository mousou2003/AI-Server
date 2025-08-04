# Churn Model Testing Guide

This guide provides comprehensive tests to verify that your custom `qwen2.5-coder:7b-churn` model is properly configured with the specialized churn analysis system prompt and behaves according to the business-focused constraints.

## Prerequisites

1. **Verify Model Availability**
   ```bash
   docker exec ollama-qwen-churn ollama list
   ```
   - Confirm `qwen2.5-coder:7b-churn` appears in the list
   - Note the creation/modification date

2. **Access Web UI**
   - Open http://localhost:3000
   - Select the `qwen2.5-coder:7b-churn` model from the dropdown
   - Ensure you're testing the custom model, not the base model

## Test Suite

### Test 1: System Prompt Integration & Role Recognition

**Test Prompt:**
```
Hello! I have a customer dataset and I'm concerned about churn. Can you help me understand what I should be looking for?
```

**Expected Response Characteristics:**
- ✅ Should identify itself as a "specialized churn analysis assistant"
- ✅ Should focus on business insights and patterns
- ✅ Should ask business-focused clarifying questions about:
  - Industry/business type
  - Customer segments
  - Available data fields
  - Current business concerns
- ✅ Should mention analyzing CSV files through conversation
- ❌ Should NOT offer to write code or provide technical implementation

**Pass Criteria:**
- Response is conversational and business-focused
- No technical jargon or programming references
- Asks relevant business questions to understand context

---

### Test 2: Code Generation Constraint Adherence

**Test Prompt:**
```
Can you write Python code to analyze my customer churn data?
```

**Expected Response:**
- ✅ Should politely decline to write code
- ✅ Should explain the conversational analysis approach
- ✅ Should redirect to discussing data patterns through dialogue
- ✅ Should maintain helpful, business-focused tone
- ❌ Should NOT provide any Python, SQL, or programming syntax
- ❌ Should NOT suggest technical tools or libraries

**Pass Criteria:**
- Clearly refuses code generation
- Offers alternative conversational analysis approach
- Stays within business analysis constraints

---

### Test 3: Business Focus & Analysis Framework

**Test Prompt:**
```
I have customer data with columns like customer_id, tenure, monthly_charges, total_charges, contract_type, and churn_status. What insights can you provide?
```

**Expected Response Structure:**
1. **Key Finding**: Should identify important business patterns from the described fields
2. **Supporting Evidence**: Should explain what these fields typically reveal (in business terms)
3. **Business Implication**: Should discuss what patterns mean for the company
4. **Recommended Action**: Should suggest specific, actionable next steps
5. **Follow-up Questions**: Should ask about business context and goals

**Expected Business Dimensions Covered:**
- Customer segments (contract types, tenure groups)
- Risk factors (charges, contract duration)
- Business impact (revenue implications)
- Timing considerations (lifecycle stages)

**Pass Criteria:**
- Uses business language, not technical terms
- Focuses on actionable insights
- Asks about business context rather than technical implementation
- Follows the 5-step response format

---

### Test 4: Technical Implementation Avoidance

**Test Prompt:**
```
Show me SQL queries to find churned customers with high monthly charges.
```

**Expected Response:**
- ✅ Should decline to provide SQL queries
- ✅ Should offer to discuss the business question behind the request
- ✅ Should ask about the business goal (why high-charge churned customers matter)
- ✅ Should suggest conversational analysis of this customer segment
- ❌ Should NOT provide any SQL syntax or database queries
- ❌ Should NOT suggest technical database tools

**Pass Criteria:**
- Refuses to provide technical implementation
- Redirects to business discussion
- Maintains focus on business value of the analysis

---

### Test 5: Statistical Formula Constraint

**Test Prompt:**
```
Calculate the churn rate formula and show me the statistical significance tests I should run.
```

**Expected Response:**
- ✅ Should explain churn rate in simple business terms (customers lost vs total customers)
- ✅ Should avoid mathematical formulas and statistical notation
- ✅ Should focus on what churn rate means for business decisions
- ✅ Should ask about business thresholds and concerns
- ❌ Should NOT provide mathematical formulas or equations
- ❌ Should NOT mention specific statistical tests or procedures

**Pass Criteria:**
- Explains concepts in plain business language
- Avoids mathematical notation and statistical jargon
- Focuses on business interpretation rather than calculation methods

---

### Test 6: Business Strategy Focus

**Test Prompt:**
```
I notice customers with month-to-month contracts have higher churn rates. What should I do about this?
```

**Expected Response Elements:**
- **Business Analysis**: Should discuss why month-to-month customers might churn more
- **Segment Understanding**: Should ask about different customer types and their needs
- **Retention Strategies**: Should suggest practical business actions like:
  - Contract incentives
  - Customer engagement programs
  - Early warning systems
  - Targeted communication
- **Risk Assessment**: Should help identify which month-to-month customers are highest risk
- **Business Impact**: Should discuss revenue implications

**Pass Criteria:**
- Provides actionable business recommendations
- Focuses on customer behavior and business strategies
- Asks relevant follow-up questions about implementation feasibility
- Avoids technical solutions or data processing details

---

### Test 7: Response Consistency Check

**Test Prompt (Repeat 3 times):**
```
What are the most important factors to monitor for early churn warning signs?
```

**Expected Consistency:**
- With temperature=0.3, responses should be fairly consistent
- Core business advice should remain stable
- Response structure should follow the 5-step format
- Should consistently avoid technical implementation details

**Pass Criteria:**
- Responses are reasonably consistent across multiple attempts
- Always maintains business focus
- Consistently follows response format guidelines

---

### Test 8: Edge Case - Direct Technical Request

**Test Prompt:**
```
Write a machine learning model to predict customer churn using scikit-learn.
```

**Expected Response:**
- ✅ Should firmly decline to provide ML code
- ✅ Should explain the business-focused conversation approach
- ✅ Should offer to discuss what factors might predict churn from a business perspective
- ✅ Should ask about business goals for churn prediction
- ❌ Should NOT mention specific ML libraries or technical approaches
- ❌ Should NOT provide any code snippets or technical guidance

**Pass Criteria:**
- Completely avoids technical implementation
- Redirects to business value discussion
- Maintains helpful, consultative tone

---

## Comparison Test with Base Model

To verify your customization is working, also test the base model `qwen2.5-coder:7b`:

**Test Prompt:**
```
I have customer churn data and need help analyzing it. What's the best approach?
```

**Base Model Expected Behavior:**
- May offer to write code or provide technical solutions
- Might suggest specific tools, libraries, or programming approaches
- Could provide more general-purpose analysis suggestions

**Custom Model Expected Behavior:**
- Should focus on business conversation and insights
- Should ask about business context and goals
- Should avoid technical implementation suggestions

## Troubleshooting

If tests fail, check:

1. **Model Selection**: Ensure you're using `qwen2.5-coder:7b-churn`, not the base model
2. **Model Creation**: Verify the custom model was created successfully
3. **Modelfile Content**: Check that the system prompt was properly integrated
4. **Container Status**: Ensure Ollama container is running and responsive

## Test Results Template

```
Date: ___________
Model: qwen2.5-coder:7b-churn

Test 1 - System Prompt Integration: ✅/❌
Test 2 - Code Generation Constraint: ✅/❌  
Test 3 - Business Focus: ✅/❌
Test 4 - Technical Implementation Avoidance: ✅/❌
Test 5 - Statistical Formula Constraint: ✅/❌
Test 6 - Business Strategy Focus: ✅/❌
Test 7 - Response Consistency: ✅/❌
Test 8 - Edge Case Technical Request: ✅/❌

Overall Assessment: ✅ PASS / ❌ FAIL

Notes:
_________________________________
_________________________________
```

## Quick Validation Checklist

✅ Model responds as churn analysis expert  
✅ Refuses code generation requests  
✅ Uses business language, not technical terms  
✅ Follows 5-step response format  
✅ Asks business-focused follow-up questions  
✅ Provides actionable recommendations  
✅ Avoids statistical formulas and technical details  
✅ Maintains conversational, helpful tone  

---

**Remember**: The goal is to ensure your custom model behaves as a business consultant, not a technical developer, when discussing customer churn analysis.
