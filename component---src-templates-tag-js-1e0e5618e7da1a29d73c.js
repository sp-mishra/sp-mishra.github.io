(window.webpackJsonp=window.webpackJsonp||[]).push([[9],{155:function(e,a,t){"use strict";t.r(a),t.d(a,"pageQuery",function(){return p});t(163);var n=t(0),r=t.n(n),l=t(4),s=t.n(l),o=t(162),c=(t(148),t(151),t(167)),i=t(164),m=t(172),d=t(169),u=function(e){var a=e.pageContext,t=e.data,n=t.allMarkdownRemark.edges,l=t.site.siteMetadata.labels;console.log(a.tag);var s=a.tag,u=t.allMarkdownRemark.totalCount,p=u+" post"+(1===u?"":"s")+' tagged with "'+s+'"';return r.a.createElement(c.a,null,r.a.createElement(i.a,{title:"Home",keywords:["gatsby","javascript","react","web development","node.js","graphql"]}),r.a.createElement("div",{className:"index-main"},r.a.createElement("div",{className:"sidebar px-4 py-2"},r.a.createElement(m.a,null)),r.a.createElement("div",{className:"post-list-main"},r.a.createElement("i",null,r.a.createElement("h2",{className:"heading"},p)),n.map(function(e){var a=e.node.frontmatter.tags;return r.a.createElement("div",{key:e.node.id,className:"container mt-5"},r.a.createElement(o.a,{to:e.node.fields.slug,className:"text-dark"},r.a.createElement("h2",{className:"heading"},e.node.frontmatter.title)),r.a.createElement("small",{className:"d-block text-info"},"Posted on ",e.node.frontmatter.date),r.a.createElement("p",{className:"mt-3 d-inline"},e.node.excerpt),r.a.createElement(o.a,{to:e.node.fields.slug,className:"text-primary"},r.a.createElement("small",{className:"d-inline-block ml-3"}," Read full post")),r.a.createElement("div",{className:"d-block"},function(e){var a=[];return e.forEach(function(e,t){l.forEach(function(n){e===n.tag&&a.push(r.a.createElement(d.a,{key:t,tag:n.tag,tech:n.tech,name:n.name,size:n.size,color:n.color}))})}),a}(a)))}))))};u.propTypes={pageContext:s.a.shape({tag:s.a.string.isRequired}),data:s.a.shape({allMarkdownRemark:s.a.shape({totalCount:s.a.number.isRequired,edges:s.a.arrayOf(s.a.shape({node:s.a.shape({frontmatter:s.a.shape({title:s.a.string.isRequired})})}).isRequired)})})};var p="4072846879";a.default=u},163:function(e,a,t){var n=t(25).f,r=Function.prototype,l=/^\s*function ([^ (]*)/;"name"in r||t(18)&&n(r,"name",{configurable:!0,get:function(){try{return(""+this).match(l)[1]}catch(e){return""}}})}}]);
//# sourceMappingURL=component---src-templates-tag-js-1e0e5618e7da1a29d73c.js.map