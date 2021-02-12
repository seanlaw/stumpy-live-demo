#!/usr/bin/env python

import pandas as pd
import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Range1d, Slider, Button, TextInput, LabelSet, Circle, HoverTool, TapTool, OpenURL, Div

from stumpy import core

class DASHBOARD():
    def __init__(self):
        self.sizing_mode='stretch_both'
        self.window = None
        self.m = None

        self.df = None
        self.ts_cds = None
        self.quad_cds = None
        self.pattern_match_cds = None
        self.guage_cds = None
        self.circle_cds = None

        self.ts_plot = None
        self.mp_plot = None
        self.pm_plot = None
        self.gauge_plot = None
        self.logo_div = None
        self.heroku_div = None

        self.slider = None
        self.play_btn = None
        self.txt_inp = None
        self.pattern_btn = None
        self.match_btn = None
        self.gauge_btn = None
        self.reset_btn = None
        self.idx = None

        self.animate_id = None

    def get_df_from_file(self):
        raw_df = pd.read_csv('raw.csv')

        mp_df = pd.read_csv('matrix_profile.csv')

        self.window = raw_df.shape[0] - mp_df.shape[0] + 1
        self.m = raw_df.shape[0] - mp_df.shape[0] + 1

        df = pd.merge(raw_df, mp_df, left_index=True, how='left', right_index=True)

        return df.reset_index()

    def get_ts_dict(self, df):
        return self.df.to_dict(orient='list')

    def get_circle_dict(self, df):
        return self.df[['index', 'y']].to_dict(orient='list')

    def get_quad_dict(self, df, pattern_idx=0, match_idx=None):
        if match_idx is None:
            match_idx = df.loc[pattern_idx, 'idx'].astype(int)
        quad_dict = dict(
            pattern_left=[pattern_idx], 
            pattern_right=[pattern_idx+self.window-1], 
            pattern_top=[max(df['y'])], 
            pattern_bottom=[0],
            match_left=[match_idx], 
            match_right=[match_idx+self.window-1], 
            match_top=[max(df['y'])], 
            match_bottom=[0],
            vert_line_left=[pattern_idx-5], 
            vert_line_right=[pattern_idx+5], 
            vert_line_top=[max(df['distance'])], 
            vert_line_bottom=[0],
            hori_line_left=[0], 
            hori_line_right=[max(df['index'])], 
            hori_line_top=[df.loc[pattern_idx, 'distance']-0.01], 
            hori_line_bottom=[df.loc[pattern_idx, 'distance']+0.01],
        )
        return quad_dict

    def get_custom_quad_dict(self, df, pattern_idx=0, match_idx=None):
        if match_idx is None:
            match_idx = df.loc[pattern_idx, 'idx'].astype(int)
        quad_dict = dict(
            pattern_left=[pattern_idx], 
            pattern_right=[pattern_idx+self.window-1], 
            pattern_top=[max(df['y'])], 
            pattern_bottom=[0],
            match_left=[match_idx], 
            match_right=[match_idx+self.window-1], 
            match_top=[max(df['y'])], 
            match_bottom=[0],
            vert_line_left=[match_idx-5], 
            vert_line_right=[match_idx+5], 
            vert_line_top=[max(df['distance'])], 
            vert_line_bottom=[0],
            hori_line_left=[0], 
            hori_line_right=[max(df['index'])], 
            hori_line_top=[df.loc[match_idx, 'distance']-0.01], 
            hori_line_bottom=[df.loc[match_idx, 'distance']+0.01],
        )
        return quad_dict

    def get_pattern_match_dict(self, df, pattern_idx=0, match_idx=None):
        if match_idx is None:
            match_idx = df['idx'].loc[pattern_idx].astype(int)
        pattern_match_dict = dict(
            index=range(self.window),
            pattern=df['y'].loc[pattern_idx:pattern_idx+self.window-1],
            match=df['y'].loc[match_idx:match_idx+self.window-1]
        )

        return pattern_match_dict

    def get_gauge_dict(self, df, pattern_idx=0):
        dist = df['distance']
        max_dist = dist.max()
        min_dist = dist.min()
        x_offset = self.df.shape[0] - self.window/2
        y_offset = max_dist/2
        distance = dist.loc[pattern_idx]
        end_angle = 225 - ((225+45)*distance/max_dist)
        text = distance.round(1).astype(str)
        gauge_dict = dict(
            x=[0+x_offset], 
            y=[0+y_offset],
            start_angle=[225],
            end_angle=[end_angle.round(0).astype(str)],
            text=[text]
        ) 

        return gauge_dict

    def get_ts_plot(self, color='black'):
        """
        Time Series Plot
        """
        ts_plot = figure(toolbar_location='above', sizing_mode=self.sizing_mode, title='Raw Time Series or Sequence', tools=['reset'])
        q = ts_plot.quad('pattern_left', 'pattern_right', 'pattern_top', 'pattern_bottom', source=self.quad_cds, name='pattern_quad', color='#54b847')
        q.visible = False
        q = ts_plot.quad('match_left', 'match_right', 'match_top', 'match_bottom', source=self.quad_cds, name='match_quad', color='#696969', alpha=0.5)
        q.visible = False
        l = ts_plot.line(x='index', y='y', source=self.ts_cds, color=color)
        ts_plot.x_range = Range1d(0, max(self.df['index']), bounds=(0, max(self.df['x'])))
        ts_plot.y_range = Range1d(0, max(self.df['y']), bounds=(0, max(self.df['y'])))

        c = ts_plot.circle(x='index', y='y', source=self.circle_cds, size=0, line_color='white')
        c.selection_glyph = Circle(line_color='white')
        c.nonselection_glyph = Circle(line_color='white')

        # Modify taptool to open URL link when point is clicked
        #url = '@url'
        #taptool = ts_plot.select(type=TapTool)
        #taptool.callback = OpenURL(url=url)

        # Add hovertool for line renderer
        #ts_plot.add_tools(HoverTool(renderers=[l], tooltips=[('Title', '@title'), ('URL', '@url'), ('Source','@domain'), ('Daily Rank', '@rank'), ('Date', '@date')]))


        return ts_plot

    def get_mp_plot(self):
        """
        Matrix Profile Plot
        """
        mp_plot = figure(x_range=self.ts_plot.x_range, toolbar_location=None, sizing_mode=self.sizing_mode, title='Matrix Profile (All Minimum Distances)')
        q = mp_plot.quad('vert_line_left', 'vert_line_right', 'vert_line_top', 'vert_line_bottom', source=self.quad_cds, name='pattern_start', color='#54b847')
        q.visible = False
        q = mp_plot.quad('hori_line_left', 'hori_line_right', 'hori_line_top', 'hori_line_bottom', source=self.quad_cds, name='match_dist', color='#696969', alpha=0.5)
        q.visible = False
        mp_plot.line(x='index', y='distance', source=self.ts_cds, color='black')
        #mp_plot.x_range = Range1d(0, self.df.shape[0]-self.window+1, bounds=(0, self.df.shape[0]-self.window+1))
        mp_plot.x_range = Range1d(0, self.df.shape[0]+1, bounds=(0, self.df.shape[0]+1))
        mp_plot.y_range = Range1d(0, max(self.df['distance']), bounds=(0, max(self.df['distance'])))

        label = LabelSet(x='x', y='y', text='text', source=self.gauge_cds, text_align='center', name='gauge_label', text_color='black', text_font_size='30pt')
        mp_plot.add_layout(label)

        return mp_plot

    def get_gauge_plot(self):
        gauge_plot = figure(toolbar_location=None, title='Distance to Closest Pattern Match')
        gauge_plot.toolbar.active_drag = None
        gauge_plot.axis.visible = False
        gauge_plot.grid.visible = False
        aw = gauge_plot.annular_wedge(
            x=[0], 
            y=[0],
            start_angle=225, 
            end_angle=-45, 
            direction='clock',
            inner_radius=0.1, 
            outer_radius=0.15,
            start_angle_units='deg',
            end_angle_units='deg',
            fill_color=None,
            line_color='#54b847',
            name='gauge_outline',
        )
        aw.visible = False
        aw = gauge_plot.annular_wedge(
            x='x', 
            y='y',
            start_angle='start_angle', 
            end_angle='end_angle', 
            source=self.gauge_cds,
            direction='clock',
            inner_radius=0.1, 
            outer_radius=0.15,
            start_angle_units='deg',
            end_angle_units='deg',
            color='#696969',
            alpha=0.5,
            name='gauge_fill',
        )
        aw.visible = False
        label = LabelSet(x='x', y='y', text='text', source=self.gauge_cds, text_align='center', y_offset=-10, name='gauge_label', text_color='white')
        gauge_plot.add_layout(label)
        return gauge_plot

    def get_pm_plot(self):
        """
        Pattern-Match Plot
        """
        pm_plot = figure(toolbar_location=None, sizing_mode=self.sizing_mode, title='Pattern Match Overlay')
        l = pm_plot.line('index', 'pattern', source=self.pattern_match_cds, name='pattern_line', color='#54b847', line_width=2)
        l.visible = False
        l = pm_plot.line('index', 'match', source=self.pattern_match_cds, name='match_line', color='#696969', alpha=0.5, line_width=2)
        l.visible = False

        return pm_plot

    def get_logo_div(self):
        """
        STUMPY logo
        """

        logo_div = Div(text="<a href='https://stumpy.readthedocs.io/en/latest/'><img src='https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/images/stumpy_logo_small.png'></a>")

        return logo_div

    def get_heroku_div(self):
        """
        STUMPY Heroku App Link
        """

        heroku_div = Div(text="http://tiny.cc/stumpy-demo")

        return heroku_div

    def get_slider(self, value=0):
        slider = Slider(start=0.0, end=max(self.df['index'])-self.window, value=value, step=1, title="Subsequence", sizing_mode=self.sizing_mode)
        return slider

    def get_play_button(self):
        play_btn = Button(label='► Play')
        play_btn.on_click(self.animate)
        return play_btn

    def get_text_input(self):
        txt_inp = TextInput(sizing_mode=self.sizing_mode)
        return txt_inp

    def get_buttons(self):
        pattern_btn = Button(label='Show Motif', sizing_mode=self.sizing_mode)
        match_btn = Button(label='Show Nearest Neighbor', sizing_mode=self.sizing_mode)
        gauge_btn = Button(label='Show Gauge', sizing_mode=self.sizing_mode)
        reset_btn = Button(label='Reset', sizing_mode=self.sizing_mode) 
        return pattern_btn, match_btn, gauge_btn, reset_btn

    def update_plots(self, attr, new, old):
        self.quad_cds.data = self.get_quad_dict(self.df, self.slider.value)
        self.pattern_match_cds.data = self.get_pattern_match_dict(self.df, self.slider.value)
        self.gauge_cds.data = self.get_gauge_dict(self.df, self.slider.value)

    def custom_update_plots(self, attr, new, old):
        self.quad_cds.data = self.get_custom_quad_dict(self.df, self.pattern_idx, self.slider.value)
        self.pattern_match_cds.data = self.get_pattern_match_dict(self.df, self.pattern_idx, self.slider.value)
        self.gauge_cds.data = self.get_gauge_dict(self.df, self.slider.value)

        dist = self.df['distance'].loc[self.slider.value]
        if dist < 15.0:
            gauge_fill = self.gauge_plot.select(name='gauge_fill')[0]
            gauge_fill.glyph.fill_color = 'red'
            gauge_fill.glyph.fill_alpha = 1.0
        else:
            gauge_fill = self.gauge_plot.select(name='gauge_fill')[0]
            gauge_fill.glyph.fill_color = '#696969'
            gauge_fill.glyph.fill_alpha = 0.5

    def show_hide_pattern(self):
        pattern_quad = self.ts_plot.select(name='pattern_quad')[0]
        pattern_start = self.mp_plot.select(name='pattern_start')[0]
        pattern_line = self.pm_plot.select(name='pattern_line')[0]
        if pattern_quad.visible:
            pattern_start.visible = False
            pattern_line.visible = False
            pattern_quad.visible = False
            self.pattern_btn.label = 'Show Motif'
        else:
            pattern_start.visible = True
            pattern_line.visible = True
            pattern_quad.visible = True
            self.pattern_btn.label = 'Hide Motif'

    def show_hide_match(self):
        match_quad = self.ts_plot.select(name='match_quad')[0]
        match_dist = self.mp_plot.select(name='match_dist')[0]
        match_line = self.pm_plot.select(name='match_line')[0]
        if match_quad.visible:
            match_dist.visible = False
            match_line.visible = False
            match_quad.visible = False
            self.match_btn.label = 'Show Nearest Neighbor'
        else:
            match_dist.visible = True
            match_line.visible = True
            match_quad.visible = True
            self.match_btn.label = 'Hide Nearest Neighbor'

    def show_hide_gauge(self):
        gauge_fill = self.gauge_plot.select(name='gauge_fill')[0]
        gauge_outline = self.gauge_plot.select(name='gauge_outline')[0]
        gauge_label = self.gauge_plot.select(name='gauge_label')[0]
        if gauge_fill.visible:
            gauge_fill.visible = False
            gauge_outline.visible = False
            gauge_label.text_color = 'white'
            self.gauge_btn.label = 'Show Gauge'
        else:        
            gauge_fill.visible = True
            gauge_outline.visible = True
            gauge_label.text_color = 'black'
            self.gauge_btn.label = 'Hide Gauge'

    def update_slider(self, attr, old, new):
        self.slider.value = int(self.txt_inp.value)

    def animate(self):
        if self.play_btn.label == '► Play':
            self.play_btn.label = '❚❚ Pause'
            self.animate_id = curdoc().add_periodic_callback(self.update_animate, 50)
        else:
            self.play_btn.label = '► Play'
            curdoc().remove_periodic_callback(self.animate_id)

    def update_animate(self, shift=50):
        if self.window < self.m:  # Probably using box select
            start = self.slider.value
            end = start + shift
            if self.df.loc[start:end, 'distance'].min() <= 15:
                self.slider.value = self.df.loc[start:end, 'distance'].idxmin()
                self.animate()
            elif self.slider.value + shift <= self.slider.end:
                self.slider.value = self.slider.value + shift
            else:
                self.slider.value = 0
        elif self.slider.value + shift <= self.slider.end:
            self.slider.value = self.slider.value + shift
        else:
            self.slider.value = 0

    def reset(self):
        self.sizing_mode='stretch_both'
        self.window = self.m

        self.default_idx = 640
        self.df = self.get_df_from_file()
        self.ts_cds.data = self.get_ts_dict(self.df)
        self.mp_plot.y_range.end = max(self.df['distance'])
        self.mp_plot.title.text = 'Matrix Profile (All Minimum Distances)'
        self.mp_plot.y_range.bounds = (0, max(self.df['distance']))
        self.quad_cds.data = self.get_quad_dict(self.df, pattern_idx=self.default_idx)
        self.pattern_match_cds.data = self.get_pattern_match_dict(self.df, pattern_idx=self.default_idx)
        self.gauge_cds.data = self.get_gauge_dict(self.df, pattern_idx=self.default_idx)
        gauge_fill = self.gauge_plot.select(name='gauge_fill')[0]
        gauge_fill.glyph.fill_color = '#696969'
        gauge_fill.glyph.fill_alpha = 0.5
        self.circle_cds.data = self.get_circle_dict(self.df)
        # Remove callback and add old callback
        if self.custom_update_plots in self.slider._callbacks['value']:
            self.slider.remove_on_change('value', self.custom_update_plots)
            self.slider.on_change('value', self.update_plots)
        self.slider.end = self.df.shape[0] - self.window
        self.slider.value = self.default_idx

    def get_data(self):
        self.default_idx = 640
        self.df = self.get_df_from_file()
        self.ts_cds = ColumnDataSource(self.get_ts_dict(self.df))
        self.quad_cds = ColumnDataSource(self.get_quad_dict(self.df, pattern_idx=self.default_idx))
        self.pattern_match_cds = ColumnDataSource(self.get_pattern_match_dict(self.df, pattern_idx=self.default_idx))
        self.gauge_cds = ColumnDataSource(self.get_gauge_dict(self.df, pattern_idx=self.default_idx))
        self.circle_cds = ColumnDataSource(self.get_circle_dict(self.df))

    def get_plots(self, ts_plot_color='black'):
        self.ts_plot = self.get_ts_plot(color=ts_plot_color)
        self.mp_plot = self.get_mp_plot()
        self.pm_plot = self.get_pm_plot()
        self.gauge_plot = self.get_gauge_plot()

    def get_widgets(self):
        self.slider = self.get_slider(value=self.default_idx)
        self.play_btn = self.get_play_button()
        self.txt_inp = self.get_text_input()
        self.pattern_btn, self.match_btn, self.gauge_btn, self.reset_btn = self.get_buttons()
        self.logo_div = self.get_logo_div()
        self.heroku_div = self.get_heroku_div()

    def set_callbacks(self):
        self.slider.on_change('value', self.update_plots)
        self.pattern_btn.on_click(self.show_hide_pattern)
        self.match_btn.on_click(self.show_hide_match)
        self.gauge_btn.on_click(self.show_hide_gauge)
        self.reset_btn.on_click(self.reset)
        self.txt_inp.on_change('value', self.update_slider)

    def get_layout(self):
        self.get_data()
        self.get_plots()
        self.get_widgets()
        self.set_callbacks()

        l = layout([
            [self.ts_plot], 
            #[self.gauge_plot, self.mp_plot], 
            [self.mp_plot],
            [self.pm_plot], 
            #[self.slider, self.txt_inp],
            [self.slider],
            #[self.sms_inp, self.play_btn, self.pattern_btn, self.match_btn, self.gauge_btn, self.reset_btn]], 
            [self.pattern_btn, self.match_btn, self.play_btn, self.logo_div]],
            sizing_mode=self.sizing_mode)

        return l

    def get_raw_layout(self):
        self.get_data()
        self.get_plots(ts_plot_color='#54b847')

        l = layout([
            [self.ts_plot],  [self.mp_plot]], 
            sizing_mode=self.sizing_mode)

        return l

