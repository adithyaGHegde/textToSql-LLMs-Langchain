/*////////////////////////
	CREATE DATABASE
////////////////////////*/
CREATE DATABASE vehicle_insurance;
USE vehicle_insurance;
/*////////////////////////
	CREATING ENTITIES
////////////////////////*/
/* 1. Incident */
CREATE TABLE TEAM10_INCIDENT
(	Incident_Id INT, 
	Incident_Type VARCHAR(30),
	Incident_Date DATE NOT NULL,
	Description VARCHAR(100),
CONSTRAINT PKINCIDENT PRIMARY KEY (Incident_Id)
);
CREATE UNIQUE INDEX PKINCIDENT ON TEAM10_INCIDENT
(Incident_Id ASC);
/* 2. Customer */
CREATE TABLE TEAM10_CUSTOMER
(
	Cust_Id	INT,
	Cust_FName	VARCHAR(15) NOT NULL,
	Cust_LName	VARCHAR(15) NOT NULL,
	Cust_DOB	DATE NOT NULL,
	Cust_Gender	CHAR(2) NOT NULL,
	Cust_Address	VARCHAR(40) NOT NULL,
	Cust_Mob_Number	BIGINT NOT NULL UNIQUE,
	Cust_Email	VARCHAR(20) UNIQUE,
	Cust_Passport_Number VARCHAR(20) UNIQUE,
	Cust_Marital_Status CHAR(8),
	Cust_PPS_Number	INTEGER UNIQUE,
    CHECK (length(Cust_Mob_Number) = 10),
    CHECK (Cust_Email REGEXP '[A-Za-z0-9]*[@][A-Za-z]*[.][A-Za-z.]*'),
	CONSTRAINT PKCUSTOMER PRIMARY KEY (Cust_Id)
);
CREATE UNIQUE INDEX PKCUSTOMER ON TEAM10_CUSTOMER (CUST_ID ASC);

/* 3. Insurance Company */
CREATE TABLE TEAM10_INSURANCE_COMPANY
(			
	Company_Name		VARCHAR(25),
	Company_Address	VARCHAR(100) UNIQUE,
	Company_Contact_Number BIGINT UNIQUE,
	Company_Fax		INTEGER UNIQUE,
	Company_Email	VARCHAR(20) UNIQUE,
	Company_Website	VARCHAR(20),
	COMPANY_LOCATION VARCHAR(20),
    CHECK (length(Company_Contact_Number) = 10),
	CHECK (Company_Email REGEXP '[A-Za-z0-9]*[@][A-Za-z0-9]*[.][A-Za-z.]*'),
    CHECK (Company_Website REGEXP 'www.[A-Za-z0-9]*[.][A-Za-z.]*'),
	CONSTRAINT PKINSURANCE_COMPANY PRIMARY KEY (Company_Name)
);
CREATE UNIQUE INDEX PKINSURANCE_COMPANY ON TEAM10_INSURANCE_COMPANY (Company_Name ASC);

/* 4. Department */
CREATE TABLE TEAM10_DEPARTMENT(	
    Department_ID	CHAR(18),
    Department_Name	VARCHAR(20),
	Department_Head	CHAR(18),
	Company_Name	VARCHAR(25),
	CONSTRAINT PKDEPARTMENT PRIMARY KEY
	(Department_ID,Company_Name),
	CONSTRAINT R_56 FOREIGN KEY (Company_Name)
	REFERENCES TEAM10_INSURANCE_COMPANY(Company_Name) ON UPDATE CASCADE ON DELETE RESTRICT
);	
CREATE UNIQUE INDEX PKDEPARTMENT ON TEAM10_DEPARTMENT
(Department_ID ASC,Company_Name ASC);

/* 5. Vehicle Service */
CREATE TABLE TEAM10_VEHICLE_SERVICE
(
	Vehicle_Service_Company_Name VARCHAR(20), 
    Vehicle_Service_Address VARCHAR(100) UNIQUE,
	Vehicle_Service_Contact VARCHAR(20) UNIQUE,
	Vehicle_Service_Incharge VARCHAR(20), 
    Vehicle_Service_Type VARCHAR(20),
    Department_Id CHAR(18),
	Company_Name VARCHAR(25),
	CONSTRAINT PKVEHICLE_SERVICE PRIMARY KEY
	(Vehicle_Service_Company_Name,Department_Id),
	CONSTRAINT R_50 FOREIGN KEY (Department_Id, Company_Name) 
    REFERENCES TEAM10_DEPARTMENT (Department_ID, Company_Name) ON UPDATE CASCADE ON DELETE CASCADE
); 
CREATE UNIQUE INDEX PKVEHICLE_SERVICE ON TEAM10_VEHICLE_SERVICE
(Vehicle_Service_Company_Name ASC, Department_Id ASC);

/* 6. Vehicle */
CREATE TABLE TEAM10_VEHICLE 
(
	Vehicle_Id INT,
    Policy_Id VARCHAR(20) UNIQUE,
	Dependent_NOK_Id VARCHAR(20),
	Vehicle_Registration_Number VARCHAR(20) UNIQUE,
    Vehicle_Value INTEGER,
    Vehicle_Type VARCHAR(20),
	Vehicle_Size INTEGER,
	Vehicle_Number_Of_Seat INTEGER,
	Vehicle_Manufacturer VARCHAR(20),
    Vehicle_Engine_Number INTEGER,
	Vehicle_Chasis_Number INTEGER UNIQUE,
	Vehicle_Number VARCHAR(20),
	Vehicle_Model_Number VARCHAR(20),
    Cust_Id INT,
	CONSTRAINT PKVEHICLE PRIMARY KEY (Vehicle_Id,Cust_Id), 
    CONSTRAINT R_92 FOREIGN KEY (Cust_Id) REFERENCES TEAM10_CUSTOMER (Cust_Id) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKVEHICLE ON TEAM10_VEHICLE (Vehicle_Id ASC, Cust_Id ASC);

/* 7. Incident Report */
CREATE TABLE TEAM10_INCIDENT_REPORT
(	Incident_Report_Id INT,
	Incident_Inspector VARCHAR(20),
	Incident_Cost	INT,
	Incident_Report_Description VARCHAR(100),
	Incident_Id INT,
    Vehicle_Id INT,
	Cust_Id INT,
CONSTRAINT PKINCIDENT_REPORT PRIMARY KEY
(Incident_Report_Id, Incident_Id, Vehicle_Id, Cust_Id),
CONSTRAINT R_83 FOREIGN KEY (Incident_Id) REFERENCES
TEAM10_INCIDENT (Incident_Id) ON UPDATE CASCADE ON DELETE RESTRICT,	
CONSTRAINT R_86 FOREIGN KEY (Vehicle_Id, Cust_Id) 
REFERENCES TEAM10_VEHICLE(Vehicle_Id, Cust_Id) ON UPDATE CASCADE ON DELETE RESTRICT		
);			
CREATE UNIQUE INDEX PKINCIDENT_REPORT ON
TEAM10_INCIDENT_REPORT (Incident_Report_Id ASC,Incident_Id ASC, Vehicle_ID ASC, Cust_Id ASC);

/* 8. Premium Payment */
CREATE TABLE TEAM10_PREMIUM_PAYMENT 
(
	Premium_Payment_Id INT,
	Premium_Payment_Amount INTEGER NOT NULL,
    Premium_Payment_Schedule DATE NOT NULL,
	CONSTRAINT PKPREMIUM_PAYMENT PRIMARY KEY (Premium_Payment_Id)
);
CREATE UNIQUE INDEX PKPREMIUM_PAYMENT ON TEAM10_PREMIUM_PAYMENT
(Premium_Payment_Id ASC);

/* 9. Receipt */
CREATE TABLE TEAM10_RECEIPT 
(
	Receipt_Id INT,
	Time DATE NOT NULL,
	Cost INTEGER NOT NULL,
	CONSTRAINT PKRECEIPT PRIMARY KEY (Receipt_Id)
);
CREATE UNIQUE INDEX PKRECEIPT ON TEAM10_RECEIPT
(Receipt_Id ASC);

/* 10. Premium Payment Receipt */
CREATE TABLE TEAM10_PREMIUM_PAYMENT_RECEIPT
(
	Premium_Payment_ID INT,
    Receipt_Id INT,
    Cust_Id INT,
    Policy_Number VARCHAR(20) NOT NULL,
    CONSTRAINT PKPPRECEIPT PRIMARY KEY (Premium_Payment_ID, Receipt_Id, Cust_Id),
    CONSTRAINT R_84 FOREIGN KEY (Premium_Payment_Id) REFERENCES TEAM10_PREMIUM_PAYMENT (Premium_Payment_Id) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT R_85 FOREIGN KEY (Cust_Id) REFERENCES TEAM10_CUSTOMER (Cust_Id) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT R_855 FOREIGN KEY (Receipt_Id) REFERENCES TEAM10_RECEIPT (Receipt_Id) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKPPRECEIPT ON TEAM10_PREMIUM_PAYMENT_RECEIPT
(Receipt_Id ASC,Premium_Payment_Id ASC,Cust_Id ASC);

/* 11. Application */
CREATE TABLE TEAM10_APPLICATION 
(
	Application_Id VARCHAR(20),
	Vehicle_Id VARCHAR(20) NOT NULL,
	Application_Status CHAR(8) NOT NULL,
	Coverage VARCHAR(50) NOT NULL,
	Cust_Id INT,
	CONSTRAINT PKAPPLICATION PRIMARY KEY (Application_Id,Cust_Id),
	CONSTRAINT R_93 FOREIGN KEY (Cust_Id) REFERENCES TEAM10_CUSTOMER(Cust_Id) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKAPPLICATION ON TEAM10_APPLICATION
(Application_Id ASC, Cust_Id ASC);

/* 12. Insurance Policy */
CREATE TABLE TEAM10_INSURANCE_POLICY 
(
	Agreement_id VARCHAR(20),
    Department_Name VARCHAR(20),
	Policy_Number VARCHAR(20) UNIQUE,
	Start_Date DATE,
	Expiry_Date DATE,
	Term_Condition_Description VARCHAR(100),
	Application_Id VARCHAR(20),
	Cust_Id INT,
	CONSTRAINT PKINSURANCE_POLICY PRIMARY KEY
	(Agreement_id,Application_Id,Cust_Id),
	CONSTRAINT R_95 FOREIGN KEY (Application_Id, Cust_Id) REFERENCES TEAM10_APPLICATION (Application_Id, Cust_Id) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE UNIQUE INDEX PKINSURANCE_POLICY ON TEAM10_INSURANCE_POLICY
(Agreement_id ASC,Application_Id ASC,Cust_Id ASC);

/* 13. Product */
CREATE TABLE TEAM10_PRODUCT
(
	Product_Number INT,
    Company_Name VARCHAR(25),
	Product_Price INTEGER,
    Product_Type CHAR(15),
    CONSTRAINT PKPRODUCT PRIMARY KEY (Product_Number,Company_Name),
    CONSTRAINT R_107 FOREIGN KEY (Company_Name) REFERENCES TEAM10_INSURANCE_COMPANY (Company_Name) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKPRODUCT ON TEAM10_PRODUCT
(Product_Number ASC,Company_Name ASC);

/* 14. Quote */
CREATE TABLE TEAM10_QUOTE
(
	Quote_Id INT,
	Issue_Date DATE NOT NULL,
	Valid_From_Date DATE NOT NULL,
	Valid_Till_Date DATE NOT NULL,
	Description VARCHAR(100),
	Product_Id INT NOT NULL,
	Coverage_Level VARCHAR(20) NOT NULL, 
    Application_Id VARCHAR(20), 
    Cust_Id INT,
	CONSTRAINT PKQU0TE PRIMARY KEY (Quote_Id,Application_Id,Cust_Id),
	CONSTRAINT R_94 FOREIGN KEY (Application_Id, Cust_Id) REFERENCES
	TEAM10_APPLICATION (Application_Id, Cust_Id) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT R_944 FOREIGN KEY (Product_Id) REFERENCES
	TEAM10_PRODUCT (Product_Number) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE UNIQUE INDEX PKQU0TE ON TEAM10_QUOTE
(Quote_Id ASC, Application_Id ASC, Cust_Id ASC);

/* 15. Staff */
CREATE TABLE TEAM10_STAFF
(
	Staff_Id INT,
	Staff_Fname VARCHAR(10),
	Staff_LName VARCHAR(10),
	Staff_Adress VARCHAR(100),
	Staff_Contact BIGINT,
	Staff_Gender CHAR(2),
	Staff_Marital_Status CHAR(8),
	Staff_Nationality CHAR(15),
	Staff_Qualification VARCHAR(20),
	Staff_Allowance INTEGER,
	Staff_PPS_Number VARCHAR(9),
	Company_Name VARCHAR(25),
	CONSTRAINT PKSTAFF PRIMARY KEY (Staff_Id,Company_Name),
	CONSTRAINT R_105 FOREIGN KEY (Company_Name) REFERENCES TEAM10_INSURANCE_COMPANY (Company_Name) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKSTAFF ON TEAM10_STAFF
(Staff_Id ASC,Company_Name ASC);

/* 16. NoK */
CREATE TABLE TEAM10_NOK 
(
	Nok_Id VARCHAR(20),
	Nok_Name VARCHAR(20),
	Nok_Address VARCHAR(100),
	Nok_Phone_Number BIGINT,
	Nok_Gender CHAR(2),
	Nok_Marital_Status CHAR(8),
	Agreement_id VARCHAR(20),
	Application_Id VARCHAR(20),
	Cust_Id INT,
	CONSTRAINT PKNOK PRIMARY KEY
	(Nok_Id,Agreement_id,Application_Id,Cust_Id),
	CONSTRAINT R_99 FOREIGN KEY (Agreement_id, Application_Id, Cust_Id)
	REFERENCES TEAM10_INSURANCE_POLICY (Agreement_id, Application_Id, Cust_Id) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKNOK ON TEAM10_NOK
(Nok_Id ASC,Agreement_id ASC,Application_Id ASC);

/* 17. Office */
CREATE TABLE TEAM10_OFFICE
(
	Office_Name VARCHAR(20),
	Office_Leader VARCHAR(20) NOT NULL,
	Contact_Information VARCHAR(20) NOT NULL,
	Address VARCHAR(100) NOT NULL,
	Admin_Cost INTEGER,
    Department_ID	CHAR(18),
	Company_Name VARCHAR(25),
	CONSTRAINT PKOFFICE PRIMARY KEY
	(Office_Name,Department_ID,Company_Name),
	CONSTRAINT R_104 FOREIGN KEY (Department_ID, Company_Name)
	REFERENCES TEAM10_DEPARTMENT (Department_ID, Company_Name) ON UPDATE CASCADE 
);
CREATE UNIQUE INDEX PKOFFICE ON TEAM10_OFFICE
(Office_Name ASC,Department_ID ASC,Company_Name ASC);

/* 18. Coverage*/
CREATE TABLE TEAM10_COVERAGE (
	Coverage_Id VARCHAR(20),
	Coverage_Amount INTEGER NOT NULL,
	Coverage_Type VARCHAR(50) NOT NULL,
	Coverage_Level CHAR(15) NOT NULL,
	Product_Id INT NOT NULL,
	Coverage_Description VARCHAR(100),
	Covearge_Terms VARCHAR(50),
	Company_Name VARCHAR(25),
	CONSTRAINT PKCOVERAGE PRIMARY KEY (Coverage_Id,Company_Name),
	CONSTRAINT R_102 FOREIGN KEY (Company_Name) REFERENCES TEAM10_INSURANCE_COMPANY (Company_Name) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT R_1022 FOREIGN KEY (Product_Id) REFERENCES TEAM10_Product (Product_Number) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKCOVERAGE ON TEAM10_COVERAGE
(Coverage_Id ASC,Company_Name ASC);

/* 19. Insurance Policy Coverage*/
CREATE TABLE TEAM10_INSURANCE_POLICY_COVERAGE (
	Agreement_id VARCHAR(20),
	Application_Id VARCHAR(20),
	Cust_Id INT,
	Coverage_Id VARCHAR(20),
	Company_Name VARCHAR(25),
	CONSTRAINT PKINSURANCE_POLICY_COVERAGE PRIMARY KEY
	(Agreement_id,Application_Id,Cust_Id,Coverage_Id,Company_Name),
	CONSTRAINT R_97 FOREIGN KEY (Agreement_id, Application_Id, Cust_Id)
	REFERENCES TEAM10_INSURANCE_POLICY (Agreement_id, Application_Id, Cust_Id) ON UPDATE CASCADE ON DELETE CASCADE,
	CONSTRAINT R_98 FOREIGN KEY (Coverage_Id, Company_Name)
	REFERENCES TEAM10_COVERAGE (Coverage_Id, Company_Name) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKINSURANCE_POLICY_COVERAGE ON TEAM10_INSURANCE_POLICY_COVERAGE
(Agreement_id ASC,Application_Id ASC,Cust_Id ASC,Coverage_Id ASC,Company_Name ASC);

/* 20. Claim*/
CREATE TABLE TEAM10_CLAIM
(
	Claim_Id INT,
	Agreement_Id VARCHAR(20) NOT NULL,
    Application_Id VARCHAR(20) NOT NULL,
	Claim_Amount INTEGER NOT NULL,
	Incident_Id INT NOT NULL,
	Damage_Type VARCHAR(20) NOT NULL,
    Date_Of_Claim DATE NOT NULL,
	Claim_Status CHAR(10) NOT NULL,
	Cust_Id INT,
	CONSTRAINT PKCLAIM PRIMARY KEY (Claim_Id,Cust_Id),
	CONSTRAINT R_88 FOREIGN KEY (Agreement_id,Application_Id,Cust_Id) REFERENCES TEAM10_INSURANCE_POLICY (Agreement_id,Application_Id,Cust_Id) ON UPDATE CASCADE,
    CONSTRAINT R_888 FOREIGN KEY (Incident_Id) REFERENCES TEAM10_INCIDENT (Incident_Id) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE UNIQUE INDEX PKCLAIM ON TEAM10_CLAIM
(Claim_Id ASC,Cust_Id ASC);

/* 21. Claim Settlement*/
CREATE TABLE TEAM10_CLAIM_SETTLEMENT 
(
	Claim_Settlement_Id INT,
	Vehicle_Id VARCHAR(20) NOT NULL,
	Date_Settled DATE NOT NULL,
	Amount_Paid INTEGER NOT NULL,
	Coverage_Id VARCHAR(20) NOT NULL,
	Claim_Id INT,
    Cust_Id INT,
	CONSTRAINT PKCLAIM_SETTLEMENT PRIMARY KEY
	(Claim_Settlement_Id,Claim_Id,Cust_Id),
	CONSTRAINT R_90 FOREIGN KEY (Claim_Id, Cust_Id) REFERENCES TEAM10_CLAIM (Claim_Id, Cust_Id) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE UNIQUE INDEX PKCLAIM_SETTLEMENT ON TEAM10_CLAIM_SETTLEMENT
(Claim_Settlement_Id ASC,Claim_Id ASC,Cust_Id ASC);