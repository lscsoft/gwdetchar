# -*- coding: utf-8 -*-
# Copyright (C) Alex Urban (2019)
#
# This file is part of the GW DetChar python package.
#
# GW DetChar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GW DetChar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gwdetchar.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for `gwdetchar.io.html`
"""

import datetime
import os
import pytest
import shutil
import sys

from getpass import getuser
from MarkupPy import markup
from pytz import reference
from unittest import mock

from gwpy.segments import (Segment, DataQualityFlag)

from .. import html
from ..._version import __version__ as gwdetchar_version
from ...utils import parse_html

from matplotlib import use
use('Agg')

__author__ = 'Alex Urban <alexander.urban@ligo.org>'


# global test objects

VERSION = gwdetchar_version

STYLESHEETS = '\n'.join([
    '<link href="{}" rel="stylesheet" media="all" />'.format(css)
    for css in html.CSS_FILES])
SCRIPTS = '\n'.join([
    '<script src="{}" type="text/javascript"></script>'.format(js)
    for js in html.JS_FILES])

NEW_BOOTSTRAP_PAGE = """<!DOCTYPE HTML>
<html lang="en">
<head>
<meta http-equiv="refresh" content="60" />
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta content="width=device-width, initial-scale=1.0" name="viewport" />
<base href="{base}" />
%s
%s
</head>
<body>
<div class="container">
</body>
</html>""" % (STYLESHEETS, SCRIPTS)

TEST_CONFIGURATION = """[section]
key = value"""

pygments_space_span = '<span style="color: #BBB"> </span>'
pygments_output = (
    '\n'
    '<span style="color: #687822">key</span>'
    f'{pygments_space_span}'
    '<span style="color: #666">=</span>'
    f'{pygments_space_span}'
    '<span style="color: #BA2121">value</span>'
)

ABOUT = f"""<div class="row">
<div class="col-md-12">
<h2>On the command-line</h2>
<p>This page was generated with the following command-line call:</p>
<div class="highlight" style="background: #f8f8f8"><pre style="line-height: 125%;"><span></span>${pygments_space_span}gwdetchar-scattering{pygments_space_span}-i{pygments_space_span}X1
</pre></div>

<p>The install path used was <code>{sys.prefix}</code>.</p>
<h2>Configuration files</h2>
<p>The following INI-format configuration file(s) were passed on the comand-line and are reproduced here in full:</p>
<div class="highlight" style="background: #f8f8f8"><pre style="line-height: 125%;"><span></span><span style="color: #008000; font-weight: bold">[section]</span>{pygments_output}
</pre></div>

<h2 class="mt-4">Environment</h2><table class="table table-sm table-hover table-responsive" id="package-table"><caption>Table of packages installed in the production environment</caption><thead><tr><th scope="col">Name</th><th scope="col">Version</th></tr></thead><tbody><tr><td>gwdetchar</td><td>1.2.3</td></tr><tr><td>gwpy</td><td>1.0.0</td></tr></tbody></table><button class="btn btn-outline-secondary btn-table mt-2" data-table-id="package-table" data-filename="package-table.csv">Export to CSV</button>
</div>
</div>"""  # noqa: E501

ABOUT_WITH_CONFIG_LIST = f"""<div class="row">
<div class="col-md-12">
<h2>On the command-line</h2>
<p>This page was generated with the following command-line call:</p>
<div class="highlight" style="background: #f8f8f8"><pre style="line-height: 125%;"><span></span>${pygments_space_span}gwdetchar-scattering{pygments_space_span}-i{pygments_space_span}X1
</pre></div>

<p>The install path used was <code>{sys.prefix}</code>.</p>
<h2>Configuration files</h2>
<p>The following INI-format configuration file(s) were passed on the comand-line and are reproduced here in full:</p>
<div id="accordion">
<div class="card mb-1 shadow-sm">
<div class="card-header">
<a class="collapsed card-link cis-link" href="#file0" data-bs-toggle="collapse">test.ini</a>
</div>
<div id="file0" class="collapse" data-parent="#accordion">
<div class="card-body">
<div class="highlight" style="background: #f8f8f8"><pre style="line-height: 125%;"><span></span><span style="color: #008000; font-weight: bold">[section]</span>{pygments_output}
</pre></div>

</div>
</div>
</div>
</div>
<h2 class="mt-4">Environment</h2><table class="table table-sm table-hover table-responsive" id="package-table"><caption>Table of packages installed in the production environment</caption><thead><tr><th scope="col">Name</th><th scope="col">Version</th></tr></thead><tbody><tr><td>gwdetchar</td><td>1.2.3</td></tr><tr><td>gwpy</td><td>1.0.0</td></tr></tbody></table><button class="btn btn-outline-secondary btn-table mt-2" data-table-id="package-table" data-filename="package-table.csv">Export to CSV</button>
</div>
</div>"""  # noqa: E501

HTML_FOOTER = """<footer class="footer">
<div class="container">
<div class="row">
<div class="col-sm-3 icon-bar">
<a href="https://github.com/gwdetchar/gwdetchar" title="View gwdetchar-%s on GitHub" target="_blank"><i class="fas fa-code"></i></a>
<a href="https://github.com/gwdetchar/gwdetchar/issues" title="Open an issue ticket" target="_blank"><i class="fas fa-ticket-alt"></i></a>
<a href="about" title="How was this page generated?"><i class="fas fa-info-circle"></i></a>
<a href="external" title="View this page&quot;s external source"><i class="fas fa-external-link-alt"></i></a>
<a href="https://apod.nasa.gov/apod/astropix.html" title="Take a break from science" target="_blank"><i class="fas fa-heartbeat"></i></a>
</div>
<div class="col-sm-6">
<p>Created by {user} at {date}</p>
</div>
</div>
</div>
</footer>""" % (VERSION)  # noqa: E501

HTML_CLOSE = """</div>
%s
</body>
</html>""" % HTML_FOOTER

FLAG_CONTENT = """<div class="card border-warning mb-1 shadow-sm">
<div class="card-header text-white bg-warning">
<a class="collapsed card-link cis-link" href="#flag0" data-bs-toggle="collapse">X1:TEST_FLAG</a>
</div>
<div id="flag0" class="collapse" data-parent="#accordion">
<div class="card-body">{plots}
{content}
</div>
</div>
</div>"""  # noqa: E501

FLAG_HTML = FLAG_CONTENT.format(content="""<pre># seg\tstart\tstop\tduration
0\t0\t66\t66.0
</pre>""", plots='')

FLAG_HTML_WITH_PLOTS = FLAG_CONTENT.format(
    content='<pre># seg\tstart\tstop\tduration\n0\t0\t66\t66.0\n</pre>',
    plots='\n<div class="row scaffold">\n<div class="col-sm-12">\n<a '
          'href="plots/X1-TEST_FLAG-0-66.png" id="a_X1-TEST_FLAG_66" '
          'title="Known (small) and active (large) analysis segments for '
          'X1:TEST_FLAG" class="fancybox" target="_blank" data-caption='
          '"Known (small) and active (large) analysis segments for '
          'X1:TEST_FLAG" data-fancybox="gallery" data-fancybox-group="images"'
          '>\n<img id="img_X1-TEST_FLAG_66" alt="X1-TEST_FLAG-0-66.png" '
          'class="img-fluid w-100" src="plots/X1-TEST_FLAG-0-66.png" />\n</a>'
          '\n</div>\n</div>')

FLAG_HTML_NO_SEGMENTS = FLAG_CONTENT.format(
    content='<p>No segments were found.</p>', plots='')

FLAG = DataQualityFlag(known=[(0, 66)], active=[(0, 66)], name='X1:TEST_FLAG')

OMEGA_SCAFFOLD = """<div class="card card-x1">
<div class="card-header pb-0">
<h5 class="card-title"><a class="cis-link" href="https://cis.ligo.org/channel/byname/X1:STRAIN" title="CIS entry for X1:STRAIN" target="_blank">X1:STRAIN</a></h5>
</div>
<div class="card-body">
<ul class="list-group">
<li class="list-group-item">
<div class="container">
<div class="row">
<h6>
<a href="./1126259462" class="text-dark">1126259462</a>
</h6>
</div>
<div class="row scaffold">
<div class="col-sm-4">
<a href="./1126259462/plots/X1-STRAIN-qscan_whitened-1.png" id="a_X1-STRAIN_1" title="X1-STRAIN-qscan_whitened-1.png" class="fancybox" target="_blank" data-caption="X1-STRAIN-qscan_whitened-1.png" data-fancybox="gallery" data-fancybox-group="images">
<img id="img_X1-STRAIN_1" alt="X1-STRAIN-qscan_whitened-1.png" class="img-fluid w-100 lazy" data-src="./1126259462/plots/X1-STRAIN-qscan_whitened-1.png" />
</a>
</div>
<div class="col-sm-4">
<a href="./1126259462/plots/X1-STRAIN-qscan_whitened-4.png" id="a_X1-STRAIN_4" title="X1-STRAIN-qscan_whitened-4.png" class="fancybox" target="_blank" data-caption="X1-STRAIN-qscan_whitened-4.png" data-fancybox="gallery" data-fancybox-group="images">
<img id="img_X1-STRAIN_4" alt="X1-STRAIN-qscan_whitened-4.png" class="img-fluid w-100 lazy" data-src="./1126259462/plots/X1-STRAIN-qscan_whitened-4.png" />
</a>
</div>
<div class="col-sm-4">
<a href="./1126259462/plots/X1-STRAIN-qscan_whitened-16.png" id="a_X1-STRAIN_16" title="X1-STRAIN-qscan_whitened-16.png" class="fancybox" target="_blank" data-caption="X1-STRAIN-qscan_whitened-16.png" data-fancybox="gallery" data-fancybox-group="images">
<img id="img_X1-STRAIN_16" alt="X1-STRAIN-qscan_whitened-16.png" class="img-fluid w-100 lazy" data-src="./1126259462/plots/X1-STRAIN-qscan_whitened-16.png" />
</a>
</div>
</div>
</div>
</li>
</ul>
</div>
</div>"""  # noqa: E501

CMDLINE = f"""<p>This page was generated with the following command-line call:</p>
<div class="highlight" style="background: #f8f8f8"><pre style="line-height: 125%;"><span></span>${pygments_space_span}gwdetchar-conlog{pygments_space_span}-i{pygments_space_span}X1
</pre></div>

<p>The install path used was <code>{sys.prefix}.</p>"""  # noqa: E501


# -- HTML unit tests ----------------------------------------------------------

def test_fancy_plot():
    # create a dummy FancyPlot instance
    test = html.FancyPlot('test.png')
    assert test.img == 'test.png'
    assert test.caption == 'test.png'

    # check that its properties are unchanged when the argument
    # to FancyPlot() is also a FancyPlot instance
    test = html.FancyPlot(test)
    assert test.img == 'test.png'
    assert test.caption == 'test.png'


def test_finalize_static_urls(tmpdir):
    base = str(tmpdir)
    static = os.path.join(base, 'static')
    (css, js) = html.finalize_static_urls(
        static, base, html.CSS_FILES, html.JS_FILES)
    assert set(css) == set(html.CSS_FILES)
    assert set(js) == set(html.JS_FILES)
    shutil.rmtree(str(tmpdir), ignore_errors=True)


def test_new_bootstrap_page():
    base = os.path.abspath(os.path.curdir)
    page = html.new_bootstrap_page(base=base, topbtn=False, refresh=True)
    assert parse_html(str(page)) == parse_html(
        NEW_BOOTSTRAP_PAGE.format(base=base))


def test_navbar():
    navbar = html.navbar(['test'], collapse=False)
    assert parse_html(navbar) == parse_html(
        '<nav class="navbar navbar-expand-md fixed-top shadow-sm">\n'
        '<div class="container-fluid">\n<div>\n'
        '<ul class="nav navbar-nav mr-auto">\n'
        '<li class="nav-item navbar-text">\n'
        'test\n</li>\n</ul>\n</div>\n</div>\n</nav>')


def test_dropdown():
    menu = html.dropdown('test', [])
    assert parse_html(str(menu)) == parse_html(
        '<a href="#" class="nav-link dropdown-toggle" role="button" '
        'data-bs-toggle="dropdown">test</a>\n<div class="dropdown-menu '
        'dropdown-1-col shadow">\n</div>')

    menu = html.dropdown('test', ['test', '#'], active=0)
    assert parse_html(str(menu)) == parse_html(
        '<a href="#" class="nav-link dropdown-toggle" role="button" '
        'data-bs-toggle="dropdown">test</a>\n<div class="dropdown-menu '
        'dropdown-1-col shadow">\ntest\n#\n</div>')

    menu = html.dropdown('test', ['test', '#'], active=[0, 1])
    assert parse_html(str(menu)) == parse_html(
        '<a href="#" class="nav-link dropdown-toggle" role="button" '
        'data-bs-toggle="dropdown">test</a>\n<div class="dropdown-menu '
        'dropdown-1-col shadow">\ntest\n#\n</div>')

    menu = html.dropdown('<test />', ['test', '#'], active=[0, 1])
    assert parse_html(str(menu)) == parse_html(
        '<a href="#" class="nav-link dropdown-toggle" role="button" '
        'data-bs-toggle="dropdown"><test /></a>\n<div class="dropdown-menu '
        'dropdown-1-col shadow">\ntest\n#\n</div>')


def test_dropdown_link():
    page = markup.page()
    html.dropdown_link(page, None)
    assert parse_html(str(page)) == parse_html(
        '<div class="dropdown-divider"></div>')

    page = markup.page()
    html.dropdown_link(page, 'test', active=True)
    assert parse_html(str(page)) == parse_html('test')

    page = markup.page()
    html.dropdown_link(page, 'test')
    assert parse_html(str(page)) == parse_html('test')


def test_get_brand():
    ((brand, help_), class_) = html.get_brand('H1', 'Test', 0, about='about')
    assert class_ == 'navbar fixed-top navbar-expand-md navbar-h1 shadow-sm'
    assert parse_html(brand) == parse_html(
        '<div class="navbar-brand border border-white rounded">H1 Test</div>')
    assert parse_html(help_) == parse_html(
        '<ul class="nav navbar-nav">\n<li class="nav-item dropdown">\n'
        '<a class="nav-link dropdown-toggle" href="#" role="button" '
        'data-bs-toggle="dropdown">Links</a>\n<div class="dropdown-menu '
        'dropdown-menu-end shadow">\n<h6 class="dropdown-header">Internal'
        '</h6>\n<a href="about" class="dropdown-item">About this page</a>\n'
        '<div class="dropdown-divider"></div>\n<h6 class="dropdown-header">'
        'External</h6>\n<a href="https://ldas-jobs.ligo-wa.caltech.edu/'
        '~detchar/summary/day/19800106" class="dropdown-item" target="_blank">'
        'LHO Summary Pages</a>\n<a href="https://alog.ligo-wa.caltech.edu/'
        'aLOG" class="dropdown-item" target="_blank">LHO Logbook</a>\n'
        '</div>\n</li>\n</ul>')


@mock.patch(
    "gwdetchar.io.html.package_list",
    return_value=[
        {"name": "gwpy", "version": "1.0.0"},
        {"name": "gwdetchar", "version": "1.2.3"},
    ],
)
def test_about_this_page(package_list, tmpdir):
    outdir = str(tmpdir)
    config_file = os.path.join(outdir, 'test.ini')
    with open(config_file, 'w') as fobj:
        fobj.write(TEST_CONFIGURATION)
    testargs = ['/opt/bin/gwdetchar-scattering',
                '-i', 'X1']
    with mock.patch.object(sys, 'argv', testargs):
        # test with a single config file
        about = html.about_this_page(config_file)
        assert parse_html(about) == parse_html(ABOUT)
        # test with a list of config files
        about = html.about_this_page([config_file])
        assert parse_html(about) == parse_html(ABOUT_WITH_CONFIG_LIST)
    # clean up
    shutil.rmtree(outdir, ignore_errors=True)


def test_get_command_line():
    testargs = ['/opt/bin/gwdetchar-conlog',
                '--html-only',
                '-i', 'X1']
    with mock.patch.object(sys, 'argv', testargs):
        cmdline = str(html.get_command_line())
        assert (f'gwdetchar-conlog{pygments_space_span}-i'
                f'{pygments_space_span}X1') in cmdline
        assert not ('--html-only' in cmdline)
        assert 'The install path used was <code>{}</code>'.format(
            sys.prefix) in cmdline


@pytest.mark.parametrize('args, kwargs, result', [
    (('test.html', 'Test link'), {},
     '<a href="test.html" target="_blank">Test link</a>'),
    (('test.html', 'Test link'), {'class_': 'test-case'},
     '<a class="test-case" href="test.html" target="_blank">Test link</a>'),
])
def test_html_link(args, kwargs, result):
    h1 = parse_html(html.html_link(*args, **kwargs))
    h2 = parse_html(result)
    assert h1 == h2


def test_cis_link():
    h1 = parse_html(html.cis_link('X1:TEST-CHANNEL'))
    h2 = parse_html(
        '<a class="cis-link" href="https://cis.ligo.org/channel/byname/'
        'X1:TEST-CHANNEL" target="_blank" title="CIS entry for '
        'X1:TEST-CHANNEL">X1:TEST-CHANNEL</a>'
    )
    assert h1 == h2


def test_fancybox_img():
    img = html.FancyPlot('X1-TEST_AUX-test-4.png')
    out = html.fancybox_img(img)
    assert parse_html(out) == parse_html(
        '<a href="X1-TEST_AUX-test-4.png" id="a_X1-TEST_AUX_4" '
        'title="X1-TEST_AUX-test-4.png" class="fancybox" target="_blank" '
        'data-caption="X1-TEST_AUX-test-4.png" data-fancybox="gallery" '
        'data-fancybox-group="images">\n<img id="img_X1-TEST_AUX_4" '
        'alt="X1-TEST_AUX-test-4.png" class="img-fluid w-100" '
        'src="X1-TEST_AUX-test-4.png" />\n</a>')


def test_scaffold_plots():
    h1 = html.scaffold_plots([
        html.FancyPlot('X1-TEST_AUX-test-4.png'),
        html.FancyPlot('X1-TEST_AUX-test-16.png')], nperrow=2)
    assert parse_html(h1) == parse_html(
        '<div class="row scaffold">\n<div class="col-sm-6">\n'
        '<a href="X1-TEST_AUX-test-4.png" id="a_X1-TEST_AUX_4" '
        'title="X1-TEST_AUX-test-4.png" class="fancybox" target="_blank" '
        'data-caption="X1-TEST_AUX-test-4.png" data-fancybox="gallery" '
        'data-fancybox-group="images">\n'
        '<img id="img_X1-TEST_AUX_4" alt="X1-TEST_AUX-test-4.png" '
        'class="img-fluid w-100 lazy" data-src="X1-TEST_AUX-test-4.png" />\n'
        '</a>\n</div>\n<div class="col-sm-6">\n'
        '<a href="X1-TEST_AUX-test-16.png" id="a_X1-TEST_AUX_16" '
        'title="X1-TEST_AUX-test-16.png" class="fancybox" target="_blank" '
        'data-caption="X1-TEST_AUX-test-16.png" data-fancybox="gallery" '
        'data-fancybox-group="images">\n'
        '<img id="img_X1-TEST_AUX_16" alt="X1-TEST_AUX-test-16.png" '
        'class="img-fluid w-100 lazy" data-src="X1-TEST_AUX-test-16.png" />\n'
        '</a>\n</div>\n</div>')


def test_download_btn():
    page = html.download_btn([('test', 'test')])
    assert parse_html(page) == parse_html(
        '<div class="dropdown float-right d-none d-lg-block">\n'
        '<button type="button" class="btn btn-outline-secondary '
        'dropdown-toggle" data-bs-toggle="dropdown" aria-expanded="false" '
        'aria-haspopup="true">Download summary</button>\n<div '
        'class="dropdown-menu dropdown-menu-end shadow">\n<a href="test" '
        'download="test" class="dropdown-item">test</a>\n</div>\n</div>')


def test_parameter_table():
    page = html.parameter_table([('test', 'test')],
                                start=0, end=1, flag='X1:TEST')
    assert '<th scope="row">Start time (UTC)</th>' in page
    assert '<td>1980-01-06 00:00:00 (0)</td>' in page
    assert '<th scope="row">End time (UTC)</th>' in page
    assert '<td>1980-01-06 00:00:01 (1)</td>' in page
    assert '<th scope="row">State flag</th>' in page
    assert '<td><code>X1:TEST</code></td>' in page
    assert '<th scope="row">test</th>' in page
    assert '<td>test</td>' in page


def test_alert():
    page = html.alert('test')
    assert parse_html(page) == parse_html(
        '<div class="alert alert-info alert-dismissible fade show text-justify'
        ' shadow-sm">\n<button type="button" class="close" data-dismiss='
        '"alert" aria-label="Close">\n<span aria-hidden="true">&times;</span>'
        '\n</button>\ntest\n</div>')


def test_alert_with_list():
    page = html.alert(['test'])
    assert parse_html(page) == parse_html(
        '<div class="alert alert-info alert-dismissible fade show text-justify'
        ' shadow-sm">\n<button type="button" class="close" data-dismiss='
        '"alert" aria-label="Close">\n<span aria-hidden="true">&times;</span>'
        '\n</button>\n<p>test</p>\n</div>')


def test_table():
    headers = ['Test']
    data = [['test']]
    caption = 'This is a test table.'
    page = html.table(headers=headers, data=data, caption=caption, id='test')
    assert parse_html(page) == parse_html(
        '<table class="table table-sm table-hover" id="test"><caption>'
        'This is a test table.</caption><thead><tr><th scope="col">Test'
        '</th></tr></thead><tbody><tr><td>test</td></tr></tbody></table>'
        '<button class="btn btn-outline-secondary btn-table mt-2" '
        'data-table-id="test" data-filename="test.csv">Export to CSV</button>')


def test_write_flag_html():
    page = html.write_flag_html(FLAG)
    assert parse_html(str(page)) == parse_html(FLAG_HTML)

    page2 = html.write_flag_html(
        DataQualityFlag(known=[], active=[], name='X1:TEST_FLAG'))
    assert parse_html(str(page2)) == parse_html(FLAG_HTML_NO_SEGMENTS)


def test_write_flag_html_with_plots(tmpdir):
    tmpdir.mkdir('plots')
    os.chdir(str(tmpdir))
    page = html.write_flag_html(FLAG, span=Segment(0, 66), plotdir='plots')
    assert parse_html(str(page)) == parse_html(FLAG_HTML_WITH_PLOTS)
    shutil.rmtree(str(tmpdir), ignore_errors=True)


def test_scaffold_omega_scans():
    times = [1126259462]
    channel = 'X1:STRAIN'
    page = html.scaffold_omega_scans(times, channel)
    assert parse_html(page) == parse_html(OMEGA_SCAFFOLD)


def test_write_footer():
    now = datetime.datetime.now()
    tz = reference.LocalTimezone().tzname(now)
    date = now.strftime('%H:%M {} on %d %B %Y'.format(tz))
    out = html.write_footer(about='about', external='external')
    assert parse_html(str(out)) == parse_html(
        HTML_FOOTER.format(user=getuser(), date=date))
    with pytest.raises(ValueError) as exc:
        html.write_footer(link='test')
    assert 'argument must be either None or a tuple' in str(exc.value)


def test_close_page(tmpdir):
    target = os.path.join(str(tmpdir), 'test.html')
    now = datetime.datetime.now()
    tz = reference.LocalTimezone().tzname(now)
    date = now.strftime('%H:%M {} on %d %B %Y'.format(tz))
    page = html.close_page(html.markup.page(), target,
                           about='about', external='external')
    assert parse_html(str(page)) == parse_html(
        HTML_CLOSE.format(user=getuser(), date=str(date)))
    assert os.path.isfile(target)
    with open(target, 'r') as fp:
        assert fp.read() == str(page)
    shutil.rmtree(target, ignore_errors=True)


@mock.patch.object(html, "CONDA", "conda_exe")
@mock.patch("{}.Path.is_dir".format(html.Path.__module__))
@mock.patch("subprocess.check_output", return_value="{\"key\": 0}")
@pytest.mark.parametrize("isdir, cmd", [
    pytest.param(
        False,
        "{} -m pip list installed --format json".format(sys.executable),
        id="pip",
    ),
    pytest.param(
        True,
        "conda_exe list --prefix {} --json".format(sys.prefix),
        id="conda",
    ),
])
def test_package_list(check_output, is_dir, isdir, cmd):
    is_dir.return_value = isdir
    assert html.package_list() == {"key": 0}
    check_output.assert_called_with(cmd.split())


@mock.patch(
    "gwdetchar.io.html.package_list",
    return_value=[
        {"name": "gwpy", "version": "1.0.0"},
        {"name": "gwdetchar", "version": "1.2.3"},
    ],
)
def test_package_table(package_list):
    assert parse_html(
        html.package_table(class_="test", caption="Test"),
    ) == parse_html(
        '<h2 class="mt-4">Environment</h2>'
        '<table class="test" id="package-table"><caption>'
        'Test</caption><thead><tr><th scope="col">Name</th><th scope="col">'
        'Version</th></tr></thead><tbody><tr><td>gwdetchar</td><td>1.2.3</td>'
        '</tr><tr><td>gwpy</td><td>1.0.0</td></tr></tbody></table><button '
        'class="btn btn-outline-secondary btn-table mt-2" data-table-id='
        '"package-table" data-filename="package-table.csv">Export to CSV'
        '</button>'
    )
