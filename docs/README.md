### Installing MKDocs

MKDocs can be easily installed using PIP

    [sudo] pip install -U mkdocs

#### Running the Site locally

MkDocs comes with a built-in webserver that lets you preview your documentation as you work on it. We start the webserver by making sure we're in the same directory as the mkdocs.yml config file, and then running the mkdocs serve command:

    $ mkdocs serve
    Running at: http://127.0.0.1:8000/

The webserver also supports auto-reloading, and will rebuild your documentation whenever anything in the configuration file, documentation directory or theme directory changes.

#### Building the HTML site

Let's build the documentation site, this will generate the HTML and any CSS files if they may be required.

    $ mkdocs build

