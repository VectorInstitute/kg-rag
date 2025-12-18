/**
 * Main diagrams module - imports and initializes all visualizations
 */

// Simple HTML escape function to prevent XSS when inserting text as HTML
function escapeHTML(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Import diagram script path for iframe reference
export const diagramPath = './diagrams/diagram.js';

// Initialize all visualizations
export function initVisualizations() {
  // Any interactive visualizations to initialize would go here
  // The main interactive visualization is already handled in the iframe
}

/**
 * Custom Table of Contents with enhanced styling
 */

export function initCustomTOC() {
  // Wait for the DOM to be fully loaded
  const tocContainer = document.querySelector('d-contents');
  if (!tocContainer) return;

  // Apply custom styling to the TOC
  const style = document.createElement('style');
  style.textContent = `
    d-contents {
      background: transparent;
      border-radius: 0;
      box-shadow: none;
      padding: 10px 0 !important;
      width: 100% !important;
      max-width: 100% !important;
      left: 0 !important;
      right: 0 !important;
      margin-left: 0 !important;
      position: relative !important;
      top: 0 !important;
      bottom: auto !important;
      margin-bottom: 10px !important;
      border: none !important;
    }

    d-article {
      padding-left: 0 !important;
    }

    /* Remove hover effect */

    d-contents nav h4 {
      font-size: 1.4em !important;
      color: #333 !important;
      margin-bottom: 0.8em !important;
      padding-bottom: 0.25em !important;
      position: relative !important;
      font-weight: 600 !important;
      display: block !important;
      width: 100% !important;
      text-align: left !important;
    }

    d-contents .toc-line {
      display: none !important;
    }

    d-contents nav {
      display: block !important;
      width: 100% !important;
      max-width: 100% !important;
    }

    d-contents nav > div.toc-section {
      margin-bottom: 0.5em !important;
      padding: 5px 15px !important;
      transition: all 0.3s ease !important;
      border-left: none !important;
      margin-right: 0 !important;
      width: 100% !important;
      display: block !important;
      background-color: transparent !important;
      box-shadow: none !important;
    }

    d-contents nav > div:hover {
      background-color: transparent !important;
    }

    d-contents nav > div > a {
      display: block !important;
      font-size: 16px !important;
      font-weight: 600 !important;
      color: #2c3e50 !important;
      padding: 3px 0 !important;
      margin: 0 !important;
      position: relative !important;
      transition: all 0.3s ease !important;
      text-decoration: none !important;
    }

    d-contents nav > div > a:hover {
      color: #8e44ad !important;
    }

    d-contents ul {
      padding-left: 0 !important;
      margin-top: 3px !important;
      margin-bottom: 3px !important;
      margin-left: 28px !important;
      list-style: none !important;
    }

    d-contents nav ul li {
      margin-bottom: 4px !important;
      position: relative !important;
      transition: all 0.3s ease !important;
      padding: 2px 5px 2px 20px !important;
      border-radius: 0 !important;
    }

    d-contents nav ul li::before {
      content: "";
    }

    d-contents nav ul li:hover {
      background-color: transparent !important;
    }

    d-contents nav a {
      color: #555 !important;
      transition: all 0.3s ease !important;
      font-size: 14px !important;
      display: block !important;
      text-decoration: none !important;
    }

    d-contents nav ul li a {
      color: #555 !important;
    }

    d-contents nav ul li:hover a {
      color: #8e44ad !important;
    }
  `;

  document.head.appendChild(style);

  // Enhance the TOC with section markers and icons
  const tocDivs = tocContainer.querySelectorAll('nav > div');

  tocDivs.forEach((div, index) => {
    const sectionNum = index + 1;
    const link = div.querySelector('a');
    if (link) {
      const originalText = link.textContent;
      link.innerHTML = `<span style="color: #2c3e50; font-weight: bold; margin-right: 5px;">${sectionNum}.</span> ${escapeHTML(originalText)}`;
    }

    // Add toc-section class to all divs
    div.classList.add('toc-section');
  });

  // Fix the double bullet issue for sub-headings with a more aggressive approach
  const fixBullets = () => {
    // Apply a full reset to ensure correct layout
    const tocSections = tocContainer.querySelectorAll('.toc-section');
    tocSections.forEach(section => {
      section.style.display = 'block';
      section.style.width = '100%';
    });

    // Remove all default list styling from ULs
    const allULs = tocContainer.querySelectorAll('ul');
    allULs.forEach(ul => {
      ul.style.listStyleType = 'none';
      ul.style.paddingLeft = '20px';
      ul.style.marginLeft = '0';
      ul.style.marginTop = '3px';
      ul.style.marginBottom = '3px';
      ul.setAttribute('role', 'list'); // For accessibility
    });

    // Process each list item to ensure no browser default bullets
    const listItems = tocContainer.querySelectorAll('li');
    listItems.forEach(li => {
      // Apply inline styles to override any browser defaults
      li.style.listStyleType = 'none';
      li.style.display = 'block';
      li.style.marginBottom = '4px';

      // Add a class for specific targeting
      li.classList.add('custom-toc-item');

      const link = li.querySelector('a');
      if (link) {
        // Style the link directly
        link.style.fontSize = '15px';
        link.style.color = '#444';
        link.style.textDecoration = 'none';
        link.style.display = 'inline-block';
      }
    });

    // Add one more style that will override all bullets
    // Check if the style element already exists
    let extraStyle = document.getElementById('custom-toc-style');
    if (!extraStyle) {
      extraStyle = document.createElement('style');
      extraStyle.id = 'custom-toc-style';
      extraStyle.textContent = `
        d-contents li,
        d-contents ul li,
        d-contents nav ul li,
        d-contents li::marker,
        d-contents ul li::marker,
        d-contents nav ul li::marker,
        .custom-toc-item::marker {
          list-style: none !important;
          list-style-type: none !important;
          list-style-position: initial !important;
          list-style-image: initial !important;
        }

        /* Simple hover effect for links with purple color */
        d-contents a:hover {
          color: #8e44ad !important;
        }

        /* Fix for the TOC layout and positioning */
        @media (min-width: 1024px) {
          d-article {
            padding-top: 2rem !important;
          }

          d-contents + * {
            margin-top: 2rem !important;
          }
        }
      `;
      document.head.appendChild(extraStyle);
    }

    // Add classes for easier styling
    const nav = tocContainer.querySelector('nav');
    if (nav) {
      nav.classList.add('clean-nav-layout');

      // Remove any potential borders or styling
      const tocElements = tocContainer.querySelectorAll('*');
      tocElements.forEach(el => {
        el.style.border = 'none';
        el.style.borderBottom = 'none';
        el.style.borderLeft = 'none';
        el.style.borderRight = 'none';
        el.style.borderTop = 'none';
        el.style.boxShadow = 'none';
      });
    }
  };

  // Run the fix immediately and after a short delay to ensure it applies
  // after the browser has rendered the list items
  fixBullets();
  setTimeout(fixBullets, 100);
  // Try again after all resources are loaded
  window.addEventListener('load', fixBullets);
}
