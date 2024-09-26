from datetime import datetime, timedelta
from streamlit import slider, session_state


def get_date_interval_picker():

    dt1 = datetime.strptime(session_state['days'][0], '%Y/%m/%d')
    dt2 = datetime.strptime(session_state['days'][-1], '%Y/%m/%d')

    if dt1 == dt2:
        slider(f"""DataTable contains only ONE Day""", value=dt1, format="DD/MM/YY", disabled=True)
        return [dt1, dt2 + timedelta(days=1)]
    else:
        date_interval = slider(
            f"""Choose DataTable Date Interval [Format d/m/y]""",
            value=(dt1, dt2),

            format="DD/MM/YY", min_value=dt1, max_value=dt2)

        res_end = date_interval[1] + timedelta(days=1)  # add offset to end date in order to capture the last day
        return [date_interval[0], res_end]
