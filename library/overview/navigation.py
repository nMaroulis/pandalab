from streamlit import html, set_page_config, logo


def page_header(title="pandaLab"):
    set_page_config(
        page_title=title,
        layout="wide",
        menu_items={
            'About': "Developed by Nick Maroulis"
        },
        page_icon='static/ico.png'
    )
    logo('static/ico.png', icon_image='static/ico.png')
    # markdown(
    #     """<style> footer { visibility: hidden;} footer:after { content:'made by Nick Maroulis';visibility: visible;position: relative;top: 2px; }</style>""",
    #     unsafe_allow_html=True)

    # PAGE HEADER
    reduce_header_height_style = """
        <style>
            div.block-container {padding-top:2em;}
        </style>
    """
    html(reduce_header_height_style)
    return

