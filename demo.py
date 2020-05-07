#!/usr/bin/env python

from bokeh.plotting import curdoc

import dashboard

db = dashboard.DASHBOARD()
layout = db.get_layout()

curdoc().add_root(layout)