-- HR Attrition SQL Analysis
-- Author: Mallesh Putha
-- Dataset: IBM HR Analytics Employee Attrition

-- ── 1. Overall attrition rate ─────────────────────────────────
SELECT
    COUNT(*)                                          AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                 AS attrition_rate_pct
FROM hr_employees;

-- ── 2. Attrition by department ────────────────────────────────
SELECT
    Department,
    COUNT(*)                                            AS total,
    SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)   AS left_count,
    ROUND(AVG(MonthlyIncome), 0)                        AS avg_income,
    ROUND(AVG(JobSatisfaction), 2)                      AS avg_satisfaction,
    ROUND(AVG(YearsAtCompany), 1)                       AS avg_tenure,
    ROUND(
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS attrition_pct
FROM hr_employees
GROUP BY Department
ORDER BY attrition_pct DESC;

-- ── 3. Overtime impact ────────────────────────────────────────
SELECT
    OverTime,
    COUNT(*)                                            AS total,
    SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)   AS attrition_count,
    ROUND(
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS attrition_pct
FROM hr_employees
GROUP BY OverTime;

-- ── 4. Salary band analysis ───────────────────────────────────
SELECT
    CASE
        WHEN MonthlyIncome < 3000  THEN 'Under $3K'
        WHEN MonthlyIncome < 6000  THEN '$3K - $6K'
        WHEN MonthlyIncome < 10000 THEN '$6K - $10K'
        ELSE 'Over $10K'
    END                                                 AS salary_band,
    COUNT(*)                                            AS total,
    SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)   AS attrition_count,
    ROUND(
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS attrition_pct
FROM hr_employees
GROUP BY salary_band
ORDER BY attrition_pct DESC;

-- ── 5. Years at company vs attrition ─────────────────────────
SELECT
    CASE
        WHEN YearsAtCompany <= 1  THEN '0-1 years'
        WHEN YearsAtCompany <= 3  THEN '1-3 years'
        WHEN YearsAtCompany <= 5  THEN '3-5 years'
        WHEN YearsAtCompany <= 10 THEN '5-10 years'
        ELSE '10+ years'
    END                                                 AS tenure_band,
    COUNT(*)                                            AS total,
    ROUND(
        SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS attrition_pct,
    ROUND(AVG(MonthlyIncome), 0)                        AS avg_income
FROM hr_employees
GROUP BY tenure_band
ORDER BY attrition_pct DESC;

-- ── 6. High risk employees (for HR action) ────────────────────
SELECT
    EmployeeNumber,
    Department,
    JobRole,
    MonthlyIncome,
    YearsAtCompany,
    JobSatisfaction,
    OverTime,
    -- Risk score (higher = more at risk)
    (
        CASE WHEN OverTime = 'Yes'         THEN 3 ELSE 0 END +
        CASE WHEN JobSatisfaction <= 2     THEN 3 ELSE 0 END +
        CASE WHEN YearsAtCompany <= 2      THEN 2 ELSE 0 END +
        CASE WHEN MonthlyIncome < 3000     THEN 2 ELSE 0 END +
        CASE WHEN WorkLifeBalance <= 2     THEN 2 ELSE 0 END
    )                                                   AS risk_score
FROM hr_employees
WHERE Attrition = 'No'   -- current employees only
ORDER BY risk_score DESC
LIMIT 20;
