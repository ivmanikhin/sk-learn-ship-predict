HOST = "ships-db.ceb9xxeumyfk.eu-central-1.rds.amazonaws.com"
USER = "admin"
PASSWORD = "ca78lo91ps23ck"
PORT = 3306
DATABASE = "ships"

TABLE_NAME = "ships_details"
COL_NAMES = "imo_no, " \
            "name, " \
            "type, " \
            "class_society, " \
            "class, " \
            "loa, " \
            "boa, " \
            "draft, " \
            "deadweight, " \
            "displacement, " \
            "gross_tonnage, " \
            "net_tonnage, " \
            "speed, " \
            "crew, " \
            "passangers, " \
            "year, " \
            "power, " \
            "engine_num, " \
            "engine_rpm, " \
            "propulsion_type, " \
            "propulsion_num"
DF_COL_NAMES = ["IMO", "Name", "Ship type", "Classification society", "Class notation", "Length overall, m",
                "Breadth, m", "Draught, m", "Deadweight, ton", "Dry displacement, ton", "Gross tonnage", "Net tonnage",
                "Speed, kn", "Crew", "Passengers", "Year of build", "Total power, kW", "Number of main engines",
                "Main engine RPM", "Propulsion type", "Number of propulsion units"]