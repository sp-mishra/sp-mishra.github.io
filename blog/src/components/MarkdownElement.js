import React from 'react';
import { Remarkable } from 'remarkable';
import toc from 'markdown-toc';

class MarkdownElement extends React.Component {
  constructor(props) {
    super(props);
    this.md = new Remarkable({
      html:         true,        // Enable HTML tags in source
      xhtmlOut:     false,        // Use '/' to close single tags (<br />)
      breaks:       true,        // Convert '\n' in paragraphs into <br>
      // langPrefix:   'language-',  // CSS language prefix for fenced blocks
      linkify: false, // Autoconvert URL-like text to links

      // Enable some language-neutral replacement + quotes beautification
      typographer:  true,

      // Double + single quotes replacement pairs, when typographer enabled,
      // and smartquotes on. Set doubles to '«»' for Russian, '„“' for German.
      quotes: '“”‘’',

      // Highlighter function. Should return escaped HTML,
      // or '' if the source string is not changed
      highlight: function (/*str, lang*/) { return ''; }
    });
  };
  render() {
    const { text } = this.props,
      html = this.md.render(text || '');

    return (<div>
      <div dangerouslySetInnerHTML={{__html: html}} />
    </div>);
  }
}

// MarkdownElement.propTypes = {
//   text: React.PropTypes.string.isRequired
// };
//
// MarkdownElement.defaultProps = {
//   text: ''
// };

export default MarkdownElement;