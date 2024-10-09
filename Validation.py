import cx_Oracle

def Val_COM_ACQ_LUT(MPN, Supplier):
    # Database connection parameters
    dsn = cx_Oracle.makedsn("10.199.104.127", "1521", service_name="scrubbing")
    connection = cx_Oracle.connect(user="Read_Only", password="READ_ONLY", dsn=dsn)
    cursor = connection.cursor()

    # First Logic: Fetch rows where both partnumber and supplier match
    query = """
    SELECT COM_PARTNUM, cm.get_man_name(man_id) SE_SUPPLIER, com_id 
    FROM cm.xlp_se_component 
    WHERE NAN_PARTNUM = NONALPHA(:1) AND cm.get_man_name(man_id) = :2
    """
    cursor.execute(query, (MPN, Supplier))
    rows = cursor.fetchall()

    # Initialize a variable to check if the exact match was found
    exact_match_found = False

    if rows:
        # If rows are found, it means we have an exact match
        db_partnumber, db_supplier, db_com_id = rows[0]
        status = "EXACT"
        result = {
            "status": status,
            "COM_ID": db_com_id,
            "COM_PARTNUM": db_partnumber,
            "SE_SUPPLIER": db_supplier
        }
        exact_match_found = True

    if not exact_match_found:
        # Second Logic: Fetch last_com_id from alias_table
        alias_query = """
        SELECT last_com_id 
        FROM cm.tbl_part_acquisition 
        WHERE OLD_NAN_PARTNUM = NONALPHA(:1) AND cm.get_man_name(OLD_MAN_ID) = :2
        """
        cursor.execute(alias_query, (MPN, Supplier))
        alias_row = cursor.fetchone()

        if alias_row:
            last_com_id = alias_row[0]
            # Use last_com_id to get corresponding details from the first table
            fetch_com_query = """
            SELECT COM_PARTNUM, cm.get_man_name(man_id) SE_SUPPLIER, com_id 
            FROM cm.xlp_se_component 
            WHERE com_id = :1
            """
            cursor.execute(fetch_com_query, (last_com_id,))
            final_row = cursor.fetchone()
            if final_row:
                status = "ACQ"
                db_partnumber, db_supplier, db_com_id = final_row
                result = {
                    "status": status,
                    "COM_ID": db_com_id,
                    "COM_PARTNUM": db_partnumber,
                    "SE_SUPPLIER": db_supplier
                }
                exact_match_found = True
            else:
                exact_match_found = False  # No match found, continue to third logic
        else:
            exact_match_found = False  # No alias found, continue to third logic

    if not exact_match_found:
        # Third Logic: Fetch se_com_id from lut table
        lut_query = """
        SELECT se_com_id 
        FROM cm.part_lookup 
        WHERE NAN_INPUT_PART = NONALPHA(:1) AND cm.get_man_name(SE_MAN_ID) = :2
        """
        cursor.execute(lut_query, (MPN, Supplier))
        lut_row = cursor.fetchone()

        if lut_row:
            se_com_id = lut_row[0]
            
            # Use se_com_id to get corresponding details from the first table
            fetch_se_com_query = """
            SELECT COM_PARTNUM, cm.get_man_name(man_id) SE_SUPPLIER, com_id 
            FROM cm.xlp_se_component 
            WHERE com_id = :1
            """
            cursor.execute(fetch_se_com_query, (se_com_id,))
            final_se_row = cursor.fetchone()
            
            if final_se_row:
                status = "LUT"
                db_partnumber, db_supplier, db_com_id = final_se_row
                result = {
                    "status": status,
                    "COM_ID": db_com_id,
                    "COM_PARTNUM": db_partnumber,
                    "SE_SUPPLIER": db_supplier,
                }
                exact_match_found = True
            else:
                # If no matching entry found for se_com_id
                status = "NOT MATCH"
                result = {
                    "status": status,
                    "COM_ID": '',
                    "COM_PARTNUM": '',
                    "SE_SUPPLIER": ''
                }
        else:
            # If no lut entry is found, proceed to handle this scenario
            status = "NOT MATCH"
            result = {
                    "status": status,
                    "COM_ID": '',
                    "COM_PARTNUM": '',
                    "SE_SUPPLIER": ''
                }

    # Close the database connection
    cursor.close()
    connection.close()
    # Return or print the result
    return result