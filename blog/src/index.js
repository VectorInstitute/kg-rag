/**
 * Main entry point for the blog
 */

import { initVisualizations, initCustomTOC } from './diagrams/index';

// Initialize all the interactive visualizations and custom TOC when the page loads
document.addEventListener('DOMContentLoaded', () => {
  initVisualizations();
  initCustomTOC();
});
