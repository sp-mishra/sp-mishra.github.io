(window.webpackJsonp=window.webpackJsonp||[]).push([[12],{EDuE:function(e,a,t){},ccoC:function(e,a,t){"use strict";t.r(a),t.d(a,"pageQuery",(function(){return i}));t("f3/d");var n=t("q1tI"),l=t.n(n),r=t("Wbzz"),c=(t("+eM2"),t("EDuE"),t("Bl7J")),s=t("vrFN"),o=t("FRpb"),m=t("o+pQ"),i="4072846879";a.default=function(e){var a=e.pageContext,t=e.data,n=t.allMarkdownRemark.edges,i=t.site.siteMetadata.labels;console.log(a.tag);var d=a.tag,u=t.allMarkdownRemark.totalCount,E=u+" post"+(1===u?"":"s")+' tagged with "'+d+'"';return l.a.createElement(c.a,null,l.a.createElement(s.a,{title:"Home",keywords:["gatsby","javascript","react","web development","node.js","graphql"]}),l.a.createElement("div",{className:"index-main"},l.a.createElement("div",{className:"sidebar px-4 py-2"},l.a.createElement(o.a,null)),l.a.createElement("div",{className:"post-list-main"},l.a.createElement("i",null,l.a.createElement("h2",{className:"heading"},E)),n.map((function(e){var a=e.node.frontmatter.tags;return l.a.createElement("div",{key:e.node.id,className:"container mt-5"},l.a.createElement(r.Link,{to:e.node.fields.slug,className:"text-dark"},l.a.createElement("h2",{className:"heading"},e.node.frontmatter.title)),l.a.createElement("small",{className:"d-block text-info"},"Posted on ",e.node.frontmatter.date),l.a.createElement("p",{className:"mt-3 d-inline"},e.node.excerpt),l.a.createElement(r.Link,{to:e.node.fields.slug,className:"text-primary"},l.a.createElement("small",{className:"d-inline-block ml-3"}," Read full post")),l.a.createElement("div",{className:"d-block"},function(e){var a=[];return e.forEach((function(e,t){i.forEach((function(n){e===n.tag&&a.push(l.a.createElement(m.a,{key:t,tag:n.tag,tech:n.tech,name:n.name,size:n.size,color:n.color}))}))})),a}(a)))})))))}}}]);
//# sourceMappingURL=component---src-templates-tag-js-d840460592e1bcdab4f4.js.map