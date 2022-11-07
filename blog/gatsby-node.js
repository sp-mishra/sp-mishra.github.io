const path = require(`path`);
const { createFilePath } = require(`gatsby-source-filesystem`);
const _ = require("lodash");

exports.onCreateNode = ({ node, getNode, actions }) => {
  console.log("onCreateNode", "\nnode: ", node, "\ngetNode: ", getNode, "\nactions: ", actions);
  const { createNodeField } = actions;
  if (node.internal.type === `MarkdownRemark`) {
    const slug = createFilePath({ node, getNode, basePath: `pages` });
    createNodeField({
      node,
      name: `slug`,
      value: slug,
    });
  }
}

exports.createPages = ({ graphql, actions }) => {
  console.log("createPages", "\ngraphql: ", graphql, "\nactions: ", actions);
  const { createPage } = actions;
  return graphql(`
    {
      allMarkdownRemark {
        edges {
          node {
            frontmatter {
              tags
            }
            fields {
              slug
            }
          }
        }
      }
    }
  `).then(result => {
        const posts = result.data.allMarkdownRemark.edges;
        console.log("then createPages", "\nposts: ",
          JSON.stringify(posts,null, 4),
          "\nresult", JSON.stringify(result, null, 4));
        posts.forEach(({ node }) => {
          console.log("then createPages posts", "\nnode: ", JSON.stringify(node, null, 4));
          createPage({
            path: node.fields.slug,
            component: path.resolve(`./src/templates/blog-post.js`),
            context: {
              // Data passed to context is available
              // in page queries as GraphQL variables.
              slug: node.fields.slug,
            },
          });
        });

        // Tag pages:
        let tags = [];
        // Iterate through each post, putting all found tags into `tags`
        _.each(posts, edge => {
          if (_.get(edge, "node.frontmatter.tags")) {
            tags = tags.concat(edge.node.frontmatter.tags);
          }
        });

        // Eliminate duplicate tags
        tags = _.uniq(tags);

        // Make tag pages
        tags.forEach(tag => {
          createPage({
            path: `/tags/${_.kebabCase(tag)}/`,
            component: path.resolve("src/templates/tag.js"),
            context: {
              tag,
            },
          });
        });

        const postsPerPage = 3;
        const numPages = Math.ceil(posts.length / postsPerPage);

        Array.from({ length: numPages }).forEach((_, i) => {
          createPage({
            path: i === 0 ? `/` : `/${i+1}`,
            component: path.resolve("./src/templates/post-list.js"),
            context: {
              limit: postsPerPage,
              skip: i*postsPerPage, 
              numPages,
              currentPage: i+1,
            }
          });
        });
    });
}
