// Add copy button to code blocks
document.addEventListener('DOMContentLoaded', function() {
    // Add copy buttons to all code blocks
    const codeBlocks = document.querySelectorAll('pre');
    
    codeBlocks.forEach((codeBlock) => {
        // Skip if already has a copy button
        if (codeBlock.querySelector('.copy-button')) return;
        
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.title = 'Copy to clipboard';
        copyButton.innerHTML = '📋';
        
        // Position the copy button
        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.display = 'inline-block';
        wrapper.style.width = '100%';
        
        // Wrap code block and add button
        codeBlock.parentNode.insertBefore(wrapper, codeBlock);
        wrapper.appendChild(codeBlock);
        wrapper.appendChild(copyButton);
        
        // Add click event to copy code
        copyButton.addEventListener('click', async () => {
            const code = codeBlock.querySelector('code') || codeBlock;
            const textToCopy = code.innerText;
            
            try {
                await navigator.clipboard.writeText(textToCopy);
                copyButton.innerHTML = '✅';
                copyButton.title = 'Copied!';
                
                // Reset button after 2 seconds
                setTimeout(() => {
                    copyButton.innerHTML = '📋';
                    copyButton.title = 'Copy to clipboard';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
                copyButton.innerHTML = '❌';
                copyButton.title = 'Failed to copy';
            }
        });
    });
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 20,
                    behavior: 'smooth'
                });
                
                // Update URL without jumping
                history.pushState(null, null, targetId);
            }
        });
    });
    
    // Add responsive menu toggle for mobile
    const menuToggle = document.querySelector('.wy-menu-toggle');
    if (menuToggle) {
        menuToggle.addEventListener('click', function() {
            document.querySelector('.wy-nav-side').classList.toggle('shift');
        });
    }
});

// Add syntax highlighting for code blocks
if (typeof hljs !== 'undefined') {
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightBlock(block);
        });
    });
}
