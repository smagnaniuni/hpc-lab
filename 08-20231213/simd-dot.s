	.file	"simd-dot.c"
	.text
	.p2align 4
	.globl	hpc_gettime
	.type	hpc_gettime, @function
hpc_gettime:
.LFB0:
	.cfi_startproc
	endbr64
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movl	$1, %edi
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	movq	%rsp, %rsi
	call	clock_gettime@PLT
	vxorps	%xmm1, %xmm1, %xmm1
	movq	24(%rsp), %rax
	xorq	%fs:40, %rax
	vcvtsi2sdq	8(%rsp), %xmm1, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm0
	vcvtsi2sdq	(%rsp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	jne	.L6
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L6:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE0:
	.size	hpc_gettime, .-hpc_gettime
	.p2align 4
	.globl	serial_dot
	.type	serial_dot, @function
serial_dot:
.LFB29:
	.cfi_startproc
	endbr64
	testl	%edx, %edx
	jle	.L10
	leal	-1(%rdx), %ecx
	xorl	%eax, %eax
	vxorpd	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L9:
	vmovss	(%rdi,%rax,4), %xmm1
	vmulss	(%rsi,%rax,4), %xmm1, %xmm1
	movq	%rax, %rdx
	incq	%rax
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	cmpq	%rdx, %rcx
	jne	.L9
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	ret
	.p2align 4,,10
	.p2align 3
.L10:
	vxorps	%xmm0, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE29:
	.size	serial_dot, .-serial_dot
	.p2align 4
	.globl	simd_dot
	.type	simd_dot, @function
simd_dot:
.LFB30:
	.cfi_startproc
	endbr64
	vxorps	%xmm0, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE30:
	.size	simd_dot, .-simd_dot
	.p2align 4
	.globl	fill
	.type	fill, @function
fill:
.LFB31:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-32, %rsp
	subq	$64, %rsp
	vmovaps	.LC4(%rip), %xmm0
	movq	%fs:40, %rax
	movq	%rax, 56(%rsp)
	xorl	%eax, %eax
	vmovaps	%xmm0, 16(%rsp)
	vmovaps	.LC5(%rip), %xmm0
	vmovaps	%xmm0, 32(%rsp)
	testl	%edx, %edx
	jle	.L14
	movl	$0xc0000000, (%rdi)
	movl	$0x3f000000, (%rsi)
	cmpl	$1, %edx
	je	.L14
	leaq	4(%rsi), %r8
	leaq	36(%rdi), %rcx
	cmpq	%rcx, %r8
	leaq	36(%rsi), %r9
	leaq	4(%rdi), %rcx
	setnb	%r8b
	cmpq	%rcx, %r9
	leal	-2(%rdx), %eax
	setbe	%cl
	orb	%cl, %r8b
	je	.L22
	cmpl	$6, %eax
	jbe	.L22
	leal	-1(%rdx), %r10d
	vmovdqa	.LC3(%rip), %ymm1
	vmovdqa	.LC8(%rip), %ymm6
	movl	$4, %eax
	movl	%r10d, %ecx
	leaq	16(%rsp), %r9
	leaq	32(%rsp), %r8
	vmovdqa	.LC9(%rip), %ymm5
	shrl	$3, %ecx
	vmovaps	.LC10(%rip), %ymm2
	salq	$5, %rcx
	addq	$4, %rcx
	.p2align 4,,10
	.p2align 3
.L23:
	vmovdqa	%ymm1, %ymm0
	vmovaps	%ymm2, %ymm7
	vpaddd	%ymm6, %ymm1, %ymm1
	vpand	%ymm5, %ymm0, %ymm0
	vgatherdps	%ymm7, (%r9,%ymm0,4), %ymm4
	vmovaps	%ymm2, %ymm7
	vgatherdps	%ymm7, (%r8,%ymm0,4), %ymm3
	vmovups	%ymm4, (%rdi,%rax)
	vmovups	%ymm3, (%rsi,%rax)
	addq	$32, %rax
	cmpq	%rax, %rcx
	jne	.L23
	movl	%r10d, %ecx
	andl	$-8, %ecx
	leal	1(%rcx), %eax
	cmpl	%ecx, %r10d
	je	.L36
	cltq
	.p2align 4,,10
	.p2align 3
.L19:
	movq	%rax, %rcx
	andl	$3, %ecx
	vmovss	32(%rsp,%rcx,4), %xmm0
	vmovss	16(%rsp,%rcx,4), %xmm1
	vmovss	%xmm1, (%rdi,%rax,4)
	vmovss	%xmm0, (%rsi,%rax,4)
	incq	%rax
	cmpl	%eax, %edx
	jg	.L19
.L36:
	vzeroupper
.L14:
	movq	56(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L39
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L22:
	.cfi_restore_state
	leaq	2(%rax), %rcx
	movl	$1, %eax
	.p2align 4,,10
	.p2align 3
.L20:
	movq	%rax, %rdx
	andl	$3, %edx
	vmovss	32(%rsp,%rdx,4), %xmm0
	vmovss	16(%rsp,%rdx,4), %xmm1
	vmovss	%xmm1, (%rdi,%rax,4)
	vmovss	%xmm0, (%rsi,%rax,4)
	incq	%rax
	cmpq	%rax, %rcx
	jne	.L20
	jmp	.L14
.L39:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE31:
	.size	fill, .-fill
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC11:
	.string	"Usage: %s [n]\n"
.LC12:
	.string	"simd-dot.c"
.LC13:
	.string	"n > 0"
.LC14:
	.string	"size < 1024*1024*200UL"
.LC15:
	.string	"Array length = %d\n"
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC17:
	.string	"Serial: result=%f, avg. time=%f (%d runs)\n"
	.align 8
.LC18:
	.string	"SIMD  : result=%f, avg. time=%f (%d runs)\n"
	.section	.rodata.str1.1
.LC21:
	.string	"Check FAILED\n"
.LC22:
	.string	"Speedup (serial/SIMD) %f\n"
.LC23:
	.string	"0 == ret"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB32:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$104, %rsp
	.cfi_def_cfa_offset 160
	movq	%fs:40, %rax
	movq	%rax, 88(%rsp)
	xorl	%eax, %eax
	cmpl	$2, %edi
	jg	.L60
	movl	$41943040, %r12d
	movl	$10485760, %r14d
	je	.L61
.L43:
	leaq	48(%rsp), %rdi
	movq	%r12, %rdx
	movl	$32, %esi
	call	posix_memalign@PLT
	testl	%eax, %eax
	jne	.L45
	leaq	56(%rsp), %rdi
	movq	%r12, %rdx
	movl	$32, %esi
	movq	48(%rsp), %r15
	call	posix_memalign@PLT
	movl	%eax, 44(%rsp)
	testl	%eax, %eax
	jne	.L46
	movq	56(%rsp), %rbp
	movl	%r14d, %edx
	movl	$1, %edi
	leaq	.LC15(%rip), %rsi
	call	__printf_chk@PLT
	movl	%r14d, %edx
	movq	%r15, %rdi
	movl	$10, %r12d
	movq	%rbp, %rsi
	leaq	64(%rsp), %rbx
	leal	-1(%r14), %r13d
	call	fill
	movq	$0x000000000, 8(%rsp)
	.p2align 4,,10
	.p2align 3
.L48:
	movq	%rbx, %rsi
	movl	$1, %edi
	call	clock_gettime@PLT
	vxorpd	%xmm5, %xmm5, %xmm5
	xorl	%eax, %eax
	vcvtsi2sdq	64(%rsp), %xmm5, %xmm1
	vcvtsi2sdq	72(%rsp), %xmm5, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm7
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovsd	%xmm7, 16(%rsp)
	.p2align 4,,10
	.p2align 3
.L47:
	vmovss	(%r15,%rax,4), %xmm0
	vmulss	0(%rbp,%rax,4), %xmm0, %xmm0
	movq	%rax, %rdx
	incq	%rax
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm1, %xmm1
	cmpq	%rdx, %r13
	jne	.L47
	movq	%rbx, %rsi
	movl	$1, %edi
	vmovsd	%xmm1, 24(%rsp)
	call	clock_gettime@PLT
	vxorpd	%xmm6, %xmm6, %xmm6
	decl	%r12d
	vmovsd	24(%rsp), %xmm1
	vcvtsi2sdq	72(%rsp), %xmm6, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm2
	vcvtsi2sdq	64(%rsp), %xmm6, %xmm0
	vaddsd	%xmm0, %xmm2, %xmm0
	vsubsd	16(%rsp), %xmm0, %xmm0
	vaddsd	8(%rsp), %xmm0, %xmm3
	vmovsd	%xmm3, 8(%rsp)
	jne	.L48
	movl	%r14d, %edx
	movq	%rbp, %rsi
	movq	%r15, %rdi
	vmovsd	%xmm1, 16(%rsp)
	vdivsd	.LC16(%rip), %xmm3, %xmm7
	vmovsd	%xmm7, 32(%rsp)
	movl	$10, %r12d
	call	fill
	vmovsd	16(%rsp), %xmm1
	vxorpd	%xmm3, %xmm3, %xmm3
	.p2align 4,,10
	.p2align 3
.L49:
	movq	%rbx, %rsi
	movl	$1, %edi
	vmovsd	%xmm1, 24(%rsp)
	vmovsd	%xmm3, 16(%rsp)
	call	clock_gettime@PLT
	vxorpd	%xmm4, %xmm4, %xmm4
	movq	%rbx, %rsi
	movl	$1, %edi
	vcvtsi2sdq	64(%rsp), %xmm4, %xmm2
	vcvtsi2sdq	72(%rsp), %xmm4, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm0
	vaddsd	%xmm2, %xmm0, %xmm3
	vmovsd	%xmm3, 8(%rsp)
	call	clock_gettime@PLT
	vxorpd	%xmm4, %xmm4, %xmm4
	decl	%r12d
	vmovsd	16(%rsp), %xmm3
	vcvtsi2sdq	72(%rsp), %xmm4, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm2
	vmovsd	24(%rsp), %xmm1
	vcvtsi2sdq	64(%rsp), %xmm4, %xmm0
	vaddsd	%xmm0, %xmm2, %xmm0
	vsubsd	8(%rsp), %xmm0, %xmm0
	vaddsd	%xmm0, %xmm3, %xmm3
	jne	.L49
	vcvtsd2ss	%xmm1, %xmm1, %xmm2
	vmovsd	32(%rsp), %xmm1
	movl	$10, %edx
	leaq	.LC17(%rip), %rsi
	movl	$1, %edi
	movl	$2, %eax
	vcvtss2sd	%xmm2, %xmm2, %xmm0
	vmovss	%xmm2, 16(%rsp)
	vdivsd	.LC16(%rip), %xmm3, %xmm7
	vmovsd	%xmm7, 8(%rsp)
	call	__printf_chk@PLT
	vmovsd	8(%rsp), %xmm1
	vxorpd	%xmm0, %xmm0, %xmm0
	movl	$10, %edx
	leaq	.LC18(%rip), %rsi
	movl	$1, %edi
	movl	$2, %eax
	call	__printf_chk@PLT
	vmovss	16(%rsp), %xmm2
	vandps	.LC19(%rip), %xmm2, %xmm2
	vcomiss	.LC20(%rip), %xmm2
	ja	.L62
	movl	$1, %edi
	leaq	.LC22(%rip), %rsi
	movl	$1, %eax
	vmovsd	32(%rsp), %xmm7
	vdivsd	8(%rsp), %xmm7, %xmm0
	call	__printf_chk@PLT
	movq	%r15, %rdi
	call	free@PLT
	movq	%rbp, %rdi
	call	free@PLT
.L40:
	movq	88(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L63
	movl	44(%rsp), %eax
	addq	$104, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L62:
	.cfi_restore_state
	movq	stderr(%rip), %rcx
	movl	$13, %edx
	movl	$1, %esi
	leaq	.LC21(%rip), %rdi
	call	fwrite@PLT
	movl	$1, 44(%rsp)
	jmp	.L40
.L60:
	movq	(%rsi), %rcx
	movq	stderr(%rip), %rdi
	leaq	.LC11(%rip), %rdx
	movl	$1, %esi
	call	__fprintf_chk@PLT
	movl	$1, 44(%rsp)
	jmp	.L40
.L61:
	movq	8(%rsi), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movl	%eax, %r14d
	testl	%eax, %eax
	jle	.L64
	cltq
	leaq	0(,%rax,4), %r12
	cmpq	$209715199, %r12
	jbe	.L43
	leaq	__PRETTY_FUNCTION__.4441(%rip), %rcx
	movl	$209, %edx
	leaq	.LC12(%rip), %rsi
	leaq	.LC14(%rip), %rdi
	call	__assert_fail@PLT
.L46:
	leaq	__PRETTY_FUNCTION__.4441(%rip), %rcx
	movl	$213, %edx
	leaq	.LC12(%rip), %rsi
	leaq	.LC23(%rip), %rdi
	call	__assert_fail@PLT
.L45:
	leaq	__PRETTY_FUNCTION__.4441(%rip), %rcx
	movl	$211, %edx
	leaq	.LC12(%rip), %rsi
	leaq	.LC23(%rip), %rdi
	call	__assert_fail@PLT
.L64:
	leaq	__PRETTY_FUNCTION__.4441(%rip), %rcx
	movl	$205, %edx
	leaq	.LC12(%rip), %rsi
	leaq	.LC13(%rip), %rdi
	call	__assert_fail@PLT
.L63:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE32:
	.size	main, .-main
	.section	.rodata
	.type	__PRETTY_FUNCTION__.4441, @object
	.size	__PRETTY_FUNCTION__.4441, 5
__PRETTY_FUNCTION__.4441:
	.string	"main"
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	1104006501
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC3:
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.long	8
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC4:
	.long	3221225472
	.long	0
	.long	1082130432
	.long	1073741824
	.align 16
.LC5:
	.long	1056964608
	.long	0
	.long	1031798784
	.long	1056964608
	.section	.rodata.cst32
	.align 32
.LC8:
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.align 32
.LC9:
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.align 32
.LC10:
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.section	.rodata.cst8
	.align 8
.LC16:
	.long	0
	.long	1076101120
	.section	.rodata.cst16
	.align 16
.LC19:
	.long	2147483647
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC20:
	.long	925353388
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
