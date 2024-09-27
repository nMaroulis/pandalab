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
    logo('static/logo-sidebar.png', icon_image='static/ico.png')
    # html(
    #     """<style> footer { visibility: hidden;} footer:after { content:'made by Nick Maroulis';visibility: visible;position: relative;top: 2px; }</style>""")

    # PAGE HEADER
    reduce_header_height_style = """
        <style>
            div.block-container {padding-top:2.2em;}
        </style>
    """
    html(reduce_header_height_style)
    return



col_style1 = """
    <style>
    [data-testid="column"] {
        # background-color: #f9f9f9;
        box-shadow: 2px 2px 2px 2px rgba(0, 0, 0, 0.05);
        border-radius: 10px;
        padding: 22px;
        font-family: "serif";
    }
        [data-testid="column"]:hover {
            background-color: #FDFDFD;
    }
    </style>
"""

col_style2 = """
    <style>
    [data-testid="column"] {
        # background-color: #f9f9f9;
        box-shadow: 4px 4px 4px 4px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        padding: 22px;
        font-family: "serif";
    }
    [data-testid="column"]:hover {
            background-color: #FDFDFD;
            # transform: translateY(-5px); /* Added hover effect */
            # filter: brightness(0.99);
            # transition: transform 2.0s ease;
    }
    </style>
"""

col_style3 = """
<style>
[data-testid="column"] {
    background-color: #fDfDfD;
    box-shadow: 6px 6px 6px rgba(0, 0, 0, 0.3);
    border-radius: 15px;
    padding: 22px;
    font-family: "serif";
}

[data-testid="column"]:hover {
        transform: translateY(-5px); /* Added hover effect */
        filter: brightness(1.05);
        ::before{
          filter: brightness(.5);
          top: -100%;
          left: 200%;
        }
        transition: transform 0.9s ease;
}
</style>
"""


col_style4 = """
<style>
[data-testid="column"] {
    background-color: #f9f9f9;
    box-shadow: 10px 10px 23px rgba(0, 0, 0, 0.4);
    border-radius: 15px;
    padding: 22px;
    font-family: "serif";
}

[data-testid="column"]:hover {
        transform: translateY(-5px); /* Added hover effect */
        filter: brightness(1.05);
        ::before{
          filter: brightness(.5);
          top: -100%;
          left: 200%;
        }
        transition: transform 0.9s ease;
}
</style>
"""

col_style5 = """
    <style>
    [data-testid="column"] {
        # background-color: #f9f9f9;
        box-shadow: 0px 0px 4px 4px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        padding: 22px;
        font-family: "serif";
    }
    [data-testid="column"]:hover {
            background-color: #FDFDFD;
            # transform: translateY(-5px); /* Added hover effect */
            # filter: brightness(0.99);
            # transition: transform 2.0s ease;
    }
    </style>
"""
