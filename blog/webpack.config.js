const assert = require('assert')
const fs = require('fs')
const path = require("path")
const webpack = require("webpack")
const HtmlWebpackPlugin = require('html-webpack-plugin')
const CopyWebpackPlugin = require('copy-webpack-plugin')
const HtmlReplaceWebpackPlugin = require('html-replace-webpack-plugin')

function format_debate(base_indent, name) {
  // Slurp in file and split into lines
  const lines = fs.readFileSync('src/' + name + '.txt', 'utf8').trim().split('\n')

  // Parse into tree based on indentation
  const tree = []
  for (const [n, line] of lines.entries()) {
    if (line.trim().length == 0) continue
    const most = line.trimLeft()
    const indent = line.length - most.length
    assert(indent % 2 == 0)
    let kids = tree
    for (var i = 0; i < indent; i += 2) {
      assert(kids.length)
      kids = kids[kids.length - 1].kids
    }
    kids.push({n: n, text: most, kids:[]})
  }

  // Dump tree to check consistency
  function dump_tree(indent, list) {
    for (const {n, text, kids} of list) {
      console.log(indent + n + ': ' + text)
      dump_tree(indent + '  ', kids)
    }
  }
  if (0)
    dump_tree('', tree)

  // Output as html
  const output = []

  // Format a single line
  function show_line(n, line) {
    const m = line.match(/^([A-Za-z ]+):\s*(.*)$/)
    assert(m, name + ':' + n + ': Bad line: ' + line)
    kind = {Red: 'red', Blue: 'blue', Judge: 'judge', 'Note for judges': 'note', 'Wikipedia': 'wikipedia', 'Question': 'question'}[m[1]]
    assert(kind, name + ':' + n + ': Bad line: ' + line)
    return [kind, m[2]]
  }

  // Recursively display our tree
  function show_node(indent, node) {
    const [kind, line] = show_line(node.n, node.text)
    const start = indent + '<li class="' + kind + '">' + '<span>' + line + '</span>'
    if (!node.kids.length)
      output.push(start + '</li>')
    else {
      output.push(start)
      show_nodes(indent + '  ', node.kids)
      output.push(indent + '</li>')
    }
  }
  function show_nodes(indent, nodes) {
    output.push(indent + '<ol class="debate">')
    for (const node of nodes)
      show_node(indent + '  ', node)
    output.push(indent + '</ol>')
  }
  show_nodes(base_indent, tree)
  return output.join('\n')
}

module.exports = {
  entry: {
    "index": "./src/index.js",
  },
  resolve: {
    extensions: [ ".js", ".html", ".npy", ".json" ]
  },
  output: {
    path: path.join(__dirname, "public"),
    filename: "[name].bundle.js",
    chunkFilename: "[name].[id].js"
  },
  module: {
    rules: [
      {
        test: /\.(html|js)$/,
        exclude: /node_modules/,
        loader: "babel-loader",
        options: {
          presets: ["@babel/preset-env"]
        }
      },
      {
        test: /\.(npy|npc)$/,
        exclude: /node_modules/,
        loader: 'numpy-loader',
        options: {
          outputPath: 'data/'
        }
      },
      {
        test: /\.(json)$/,
        exclude: /node_modules/,
        loader: 'json-loader',
        options: {
          outputPath: 'data/'
        }
      },
      {
        test: /\.svg$/,
        exclude: /node_modules/,
        loader: 'svg-inline-loader',
        options: {
          removeSVGTagAttrs: true,
          removingTagAttrs: ["font-family"]
        }
      },
      {
        test: /\.(png|jpg|jpeg)$/,
        exclude: /node_modules/,
        loader: 'file',
        options: {
          outputPath: 'images/'
        }
      }
    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: "./src/index.ejs",
      filename: "index.html",
      chunks: ["index"]
    }),
    new CopyWebpackPlugin({
      patterns: [
        { from: 'static/' },
        { from: 'src/assets/', to: 'assets/' },
        { from: 'src/diagrams/', to: 'diagrams/' }
      ]
    }),
    new HtmlReplaceWebpackPlugin([
      {
        pattern: /<d-cite\s+key="([^"]*)"\s*\/>/g,
        replacement: (_, key) => '<d-cite key="' + key + '"></d-cite>'
      },
      {
        pattern: /<(\/?)todo>/g,
        replacement: (_, slash) => slash ? '</span>' : '<span class="todo">'
      },
      {
        pattern: /<(\/?)dtodo>/g,
        replacement: (_, slash) => slash ? '</div>' : '<div class="todo">'
      },
      {
        pattern: /( *)<debate name="([^"]+)"\/>/g,
        replacement: (_, indent, name) => format_debate(indent, name)
      },
      {
        pattern: '<script type="text/javascript" src="index.bundle.js"></script>',
        replacement: ''
      },
      {
        pattern: '</d-appendix>',
        replacement: '</d-appendix>'
      }
    ])
  ],
  devServer: {
    static: {
      directory: path.join(__dirname, 'public')
    },
    historyApiFallback: true,
    hot: true,
    client: {
      overlay: true
    },
    devMiddleware: {
      stats: "minimal"
    }
  },
  devtool: "inline-source-map"
}
